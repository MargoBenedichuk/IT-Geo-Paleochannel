# Fast SAM ONNX inference (thin, UI-friendly)
# - deps: numpy, onnxruntime, pillow
# - optional: scikit-image (remove_small_holes/remove_small_objects)
# - NO cv2, NO scipy, NO torch
#
# Input:
#   rgb: HxWx3 uint8 (RGB) OR float32 in [0..1] / [0..255]
#   points: [(x,y), ...] in ORIGINAL image pixel coords
#   labels: [1|0] same length as points (1=positive, 0=negative)
#
# Output:
#   SamResult(mask uint8 {0,1}, probs float32 [0..1] optional)
#
# Postprocess:
#   1) sigmoid if needed
#   2) resize pred -> 1024 (if required), crop padding, resize back to original
#   3) threshold -> binary
#   4) optional: keep only components connected to any positive point (BFS)
#   5) optional: skimage remove_small_holes / remove_small_objects

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


# HuggingFace SamImageProcessor defaults
_SAM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_SAM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = 1024

# default light cleanup (can override in infer())
MIN_HOLE_AREA_DEFAULT = 300
MIN_OBJECT_AREA_DEFAULT = 900


@dataclass
class SamResult:
    mask: np.ndarray                   # (H,W) uint8 {0,1}
    probs: Optional[np.ndarray] = None  # (H,W) float32 [0..1], if requested


def _get_ort_providers(prefer_cuda: bool = True) -> List[str]:
    avail = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _make_session(model_path: str, providers: List[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def _to_float01_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3, got {rgb.shape}")

    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0

    x = rgb.astype(np.float32, copy=False)
    mx = float(np.nanmax(x)) if x.size else 0.0
    if mx <= 1.0:
        return np.clip(x, 0.0, 1.0)
    return np.clip(x / 255.0, 0.0, 1.0)


def _resize_longest_to_target(rgb01: np.ndarray, target: int) -> Tuple[np.ndarray, int, int, float, float]:
    h, w = rgb01.shape[:2]
    if w >= h:
        new_w = int(target)
        new_h = int(round(target * (h / w)))
    else:
        new_h = int(target)
        new_w = int(round(target * (w / h)))

    im = Image.fromarray((rgb01 * 255.0).astype(np.uint8), mode="RGB")
    im_r = im.resize((new_w, new_h), resample=Image.BILINEAR)
    resized = np.asarray(im_r, dtype=np.float32) / 255.0

    sx = new_w / float(w)
    sy = new_h / float(h)
    return resized, new_h, new_w, sx, sy


def _make_pixel_values(rgb: np.ndarray) -> Tuple[np.ndarray, int, int, float, float]:
    rgb01 = _to_float01_rgb(rgb)
    resized, new_h, new_w, sx, sy = _resize_longest_to_target(rgb01, TARGET_SIZE)

    pad = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)
    pad[:new_h, :new_w, :] = resized

    pad = (pad - _SAM_MEAN.reshape(1, 1, 3)) / _SAM_STD.reshape(1, 1, 3)
    pixel_values = np.transpose(pad, (2, 0, 1))[None].astype(np.float32, copy=False)
    pixel_values = np.ascontiguousarray(pixel_values)
    return pixel_values, new_h, new_w, sx, sy


def _sigmoid_if_needed(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mn >= -1e-3 and mx <= 1.0 + 1e-3:
        return np.clip(x, 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-x))


def _to_hw(pred: np.ndarray) -> np.ndarray:
    m = np.asarray(pred)
    m = np.squeeze(m)
    if m.ndim == 2:
        return m
    if m.ndim == 3:
        return m[0]
    if m.ndim == 4:
        return m[0, 0]
    raise ValueError(f"Unexpected pred shape after squeeze: {m.shape}")


def _resize_float_map(map2d: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    im = Image.fromarray(map2d.astype(np.float32, copy=False), mode="F")
    im_r = im.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(im_r, dtype=np.float32)


def _bfs_keep_from_positive_points(mask01: np.ndarray, pos_points: Sequence[Tuple[int, int]], *, min_area: int = 0) -> np.ndarray:
    if mask01.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask01.shape
    if not pos_points:
        return mask01.astype(np.uint8, copy=False)

    m = mask01 > 0
    out = np.zeros((h, w), dtype=np.uint8)
    seen = np.zeros((h, w), dtype=np.uint8)
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for (x, y) in pos_points:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        if not m[yi, xi] or seen[yi, xi]:
            continue

        q = deque([(yi, xi)])
        seen[yi, xi] = 1
        comp = []

        while q:
            r, c = q.popleft()
            if not m[r, c]:
                continue
            comp.append((r, c))
            for dr, dc in neigh:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc]:
                    seen[rr, cc] = 1
                    if m[rr, cc]:
                        q.append((rr, cc))

        if min_area > 0 and len(comp) < int(min_area):
            continue

        for r, c in comp:
            out[r, c] = 1

    return out


def _apply_skimage_cleanup(mask01: np.ndarray, *, remove_holes_area: int = 0, remove_objects_min_size: int = 0) -> np.ndarray:
    """
    Optional. Lazy import. Compatible with skimage <0.26 and >=0.26 where params were renamed:
      - remove_small_holes: area_threshold -> max_size
      - remove_small_objects: min_size     -> max_size
    """
    if (remove_holes_area <= 0) and (remove_objects_min_size <= 0):
        return mask01.astype(np.uint8, copy=False)

    try:
        from skimage.morphology import remove_small_holes, remove_small_objects
    except Exception:
        return mask01.astype(np.uint8, copy=False)

    mb = (mask01 > 0)

    if remove_holes_area > 0:
        try:
            mb = remove_small_holes(mb, max_size=int(remove_holes_area))
        except TypeError:
            mb = remove_small_holes(mb, area_threshold=int(remove_holes_area))

    if remove_objects_min_size > 0:
        try:
            mb = remove_small_objects(mb, max_size=int(remove_objects_min_size))
        except TypeError:
            mb = remove_small_objects(mb, min_size=int(remove_objects_min_size))

    return mb.astype(np.uint8, copy=False)


def _infer_points_tensors(
    points: Sequence[Tuple[int, int]],
    labels: Sequence[int],
    sx: float,
    sy: float,
    fixed_n: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    if len(points) != len(labels):
        raise ValueError("points/labels length mismatch")
    if len(points) == 0:
        raise ValueError("Need at least 1 point")

    events = [(int(x), int(y), 1.0 if int(lab) == 1 else 0.0) for (x, y), lab in zip(points, labels)]

    if fixed_n is not None:
        n = int(fixed_n)
        ev = events[-n:] if len(events) >= n else events[:]
        while len(ev) < n:
            ev.append(ev[-1])
    else:
        ev = events

    pts = np.empty((len(ev), 2), dtype=np.float32)
    lbs = np.empty((len(ev),), dtype=np.float32)

    pos_pts_orig: List[Tuple[int, int]] = []
    for i, (x, y, lab) in enumerate(ev):
        if lab >= 0.5:
            pos_pts_orig.append((x, y))

        px = float(np.clip(x * sx, 0, TARGET_SIZE - 1))
        py = float(np.clip(y * sy, 0, TARGET_SIZE - 1))
        pts[i, 0] = px
        pts[i, 1] = py
        lbs[i] = float(lab)

    input_points = pts[None, None, :, :]  # (1,1,N,2)
    input_labels = lbs[None, None, :]     # (1,1,N)
    return np.ascontiguousarray(input_points), np.ascontiguousarray(input_labels), pos_pts_orig


class SamOnnxModel:
    def __init__(
        self,
        model_path: str,
        *,
        thresh: float = 0.5,
        providers: Optional[List[str]] = None,
        prefer_cuda: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX not found: {model_path}")

        if providers is None:
            providers = _get_ort_providers(prefer_cuda=prefer_cuda)

        self.thresh = float(thresh)
        self.session = _make_session(model_path, providers)

        ins = {i.name: i for i in self.session.get_inputs()}
        self.in_pixel = "pixel_values"
        self.in_points = "input_points"
        self.in_labels = "input_labels" if "input_labels" in ins else None

        if self.in_pixel not in ins or self.in_points not in ins:
            raise RuntimeError(f"Model must have inputs: '{self.in_pixel}', '{self.in_points}'. Got: {list(ins.keys())}")

        outs = self.session.get_outputs()
        if not outs:
            raise RuntimeError("Model has no outputs")
        self.out_name = outs[0].name

        shp = ins[self.in_points].shape  # (batch,1,N,2)
        fixed_n = None
        if isinstance(shp, (list, tuple)) and len(shp) == 4:
            n_dim = shp[2]
            if isinstance(n_dim, int):
                fixed_n = int(n_dim)
        self.fixed_n_points = fixed_n

    def infer(
        self,
        rgb: np.ndarray,
        points: Sequence[Tuple[int, int]],
        labels: Sequence[int],
        *,
        return_probs: bool = False,
        keep_pos_components: bool = True,
        min_area: int = 0,
        remove_holes_area: int = MIN_HOLE_AREA_DEFAULT,
        remove_objects_min_size: int = MIN_OBJECT_AREA_DEFAULT,
    ) -> SamResult:
        if len(points) == 0:
            raise ValueError("Need at least 1 point")
        if len(points) != len(labels):
            raise ValueError("points/labels length mismatch")

        orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])

        pixel_values, new_h, new_w, sx, sy = _make_pixel_values(rgb)

        ip, il, pos_pts_orig = _infer_points_tensors(
            points=points,
            labels=labels,
            sx=sx,
            sy=sy,
            fixed_n=self.fixed_n_points,
        )

        feeds = {self.in_pixel: pixel_values, self.in_points: ip}
        if self.in_labels is not None:
            feeds[self.in_labels] = il

        pred = self.session.run([self.out_name], feeds)[0]

        m2d = _to_hw(pred)
        probs = _sigmoid_if_needed(m2d)

        if probs.shape != (TARGET_SIZE, TARGET_SIZE):
            probs = _resize_float_map(probs, TARGET_SIZE, TARGET_SIZE)

        probs = probs[:new_h, :new_w]
        probs_orig = _resize_float_map(probs, orig_w, orig_h)

        mask01 = (probs_orig > self.thresh).astype(np.uint8)

        if keep_pos_components and pos_pts_orig and mask01.any():
            mask01 = _bfs_keep_from_positive_points(mask01, pos_pts_orig, min_area=int(min_area))

        if mask01.any() and ((remove_holes_area > 0) or (remove_objects_min_size > 0)):
            mask01 = _apply_skimage_cleanup(
                mask01,
                remove_holes_area=int(remove_holes_area),
                remove_objects_min_size=int(remove_objects_min_size),
            )

        return SamResult(mask=mask01, probs=(probs_orig if return_probs else None))


def run_inference(
    rgb: np.ndarray,
    points: Sequence[Tuple[int, int]],
    labels: Sequence[int],
    model: SamOnnxModel,
    *,
    return_probs: bool = False,
    keep_pos_components: bool = True,
    min_area: int = 0,
    remove_holes_area: int = MIN_HOLE_AREA_DEFAULT,
    remove_objects_min_size: int = MIN_OBJECT_AREA_DEFAULT,
) -> SamResult:
    return model.infer(
        rgb,
        points,
        labels,
        return_probs=return_probs,
        keep_pos_components=keep_pos_components,
        min_area=min_area,
        remove_holes_area=remove_holes_area,
        remove_objects_min_size=remove_objects_min_size,
    )
