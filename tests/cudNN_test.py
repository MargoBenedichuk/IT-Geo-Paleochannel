# =========================
# file: tools/check_sam_onnx_env.py
# =========================
# Purpose: one-command diagnostic for SAM ONNX on Windows
# - No cv2, no UI
# - Verifies: python path, ort providers, cudnn visibility, session providers (GPU/CPU)
# - Runs one real inference on a synthetic RGB image with a few prompts
#
# Usage (PowerShell):
#   python .\tools\check_sam_onnx_env.py --onnx "C:\path\to\SAM_river.onnx"
#
# Optional:
#   python .\tools\check_sam_onnx_env.py --onnx "...onnx" --prefer-cuda 1 --thresh 0.5

from __future__ import annotations

import os
import sys
import argparse
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


# --------- Windows DLL help (no global PATH edits needed) ----------
def _add_windows_dll_dirs() -> None:
    if os.name != "nt":
        return
    # Add common CUDA / cuDNN install dirs if present.
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.18\bin\13.1\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.18\bin\12.9\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.8",
        r"C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.7",
        r"C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.6",
        r"C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.5",
        r"C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.4",
    ]
    for p in candidates:
        try:
            if os.path.isdir(p):
                os.add_dll_directory(p)
        except Exception:
            pass


def _cmd_where(name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import subprocess
        out = subprocess.check_output(["cmd", "/c", "where", name], stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


# --------- Minimal preprocessing compatible with your SAM ONNX ----------
_SAM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_SAM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = 1024


@dataclass
class SamResult:
    mask: np.ndarray                   # (H,W) uint8 {0,1}
    probs: Optional[np.ndarray] = None  # (H,W) float32 [0..1]


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


def _resize_longest(rgb01: np.ndarray, target: int):
    h, w = rgb01.shape[:2]
    if w >= h:
        new_w = target
        new_h = int(round(target * (h / w)))
    else:
        new_h = target
        new_w = int(round(target * (w / h)))

    im = Image.fromarray((rgb01 * 255.0).astype(np.uint8), mode="RGB")
    im_r = im.resize((new_w, new_h), resample=Image.BILINEAR)
    resized = np.asarray(im_r, dtype=np.float32) / 255.0

    sx = new_w / float(w)
    sy = new_h / float(h)
    return resized, new_h, new_w, sx, sy


def _make_pixel_values(rgb: np.ndarray):
    rgb01 = _to_float01_rgb(rgb)
    resized, new_h, new_w, sx, sy = _resize_longest(rgb01, TARGET_SIZE)

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


def _get_ort_providers(prefer_cuda: bool) -> List[str]:
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


def _infer_once(
    sess: ort.InferenceSession,
    rgb: np.ndarray,
    points: Sequence[Tuple[int, int]],
    labels: Sequence[int],
    thresh: float,
) -> SamResult:
    in_names = [i.name for i in sess.get_inputs()]
    if "pixel_values" not in in_names or "input_points" not in in_names:
        raise RuntimeError(f"ONNX inputs must include pixel_values & input_points. Got: {in_names}")
    has_labels = ("input_labels" in in_names)

    pixel_values, new_h, new_w, sx, sy = _make_pixel_values(rgb)

    pts = np.empty((len(points), 2), dtype=np.float32)
    lbs = np.empty((len(points),), dtype=np.float32)
    for i, ((x, y), lab) in enumerate(zip(points, labels)):
        pts[i, 0] = float(np.clip(x * sx, 0, TARGET_SIZE - 1))
        pts[i, 1] = float(np.clip(y * sy, 0, TARGET_SIZE - 1))
        lbs[i] = 1.0 if int(lab) == 1 else 0.0

    input_points = pts[None, None, :, :]
    input_labels = lbs[None, None, :]

    feeds = {"pixel_values": pixel_values, "input_points": np.ascontiguousarray(input_points)}
    if has_labels:
        feeds["input_labels"] = np.ascontiguousarray(input_labels)

    out_name = sess.get_outputs()[0].name
    pred = sess.run([out_name], feeds)[0]

    m2d = _to_hw(pred)
    probs = _sigmoid_if_needed(m2d)
    if probs.shape != (TARGET_SIZE, TARGET_SIZE):
        probs = _resize_float_map(probs, TARGET_SIZE, TARGET_SIZE)

    probs = probs[:new_h, :new_w]
    probs_orig = _resize_float_map(probs, rgb.shape[1], rgb.shape[0])

    mask = (probs_orig > float(thresh)).astype(np.uint8)
    return SamResult(mask=mask, probs=probs_orig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to SAM ONNX model")
    ap.add_argument("--prefer-cuda", type=int, default=1, help="1 to prefer CUDA, 0 for CPU")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    _add_windows_dll_dirs()

    print("PY:", sys.executable)
    print("CWD:", os.getcwd())
    print("ONNX:", args.onnx)
    print("ORT:", ort.__version__)
    print("ORT_FILE:", ort.__file__)
    print("AVAILABLE_PROVIDERS:", ort.get_available_providers())

    if os.name == "nt":
        print("WHERE nvcc.exe:", _cmd_where("nvcc.exe") or "<not found>")
        print("WHERE cudart64_12.dll:", _cmd_where("cudart64_12.dll") or "<not found>")
        print("WHERE cudnn64_9.dll:", _cmd_where("cudnn64_9.dll") or "<not found>")

    providers = _get_ort_providers(bool(args.prefer_cuda))
    print("REQUESTED_PROVIDERS:", providers)

    try:
        sess = _make_session(args.onnx, providers)
    except Exception as e:
        print("SESSION_CREATE_ERROR:", repr(e))
        print(traceback.format_exc())
        # Try CPU fallback to still validate the ONNX itself
        try:
            sess = _make_session(args.onnx, ["CPUExecutionProvider"])
            print("FALLBACK_SESSION_PROVIDERS:", sess.get_providers())
        except Exception as e2:
            print("CPU_FALLBACK_FAILED:", repr(e2))
            return 2

    print("SESSION_PROVIDERS:", sess.get_providers())

    print("ONNX INPUTS:")
    for i in sess.get_inputs():
        print(" ", i.name, i.shape, i.type)
    print("ONNX OUTPUTS:")
    for o in sess.get_outputs():
        print(" ", o.name, o.shape, o.type)

    # Synthetic test image (deterministic)
    h, w = 512, 768
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(255, 0, w, dtype=np.uint8)[None, :]
    img[:, :, 2] = 64

    # Prompts: one positive in center, one negative nearby
    points = [(w // 2, h // 2), (w // 2 + 40, h // 2)]
    labels = [1, 0]

    try:
        res = _infer_once(sess, img, points, labels, thresh=args.thresh)
    except Exception as e:
        print("INFER_ERROR:", repr(e))
        print(traceback.format_exc())
        return 3

    m = res.mask
    p = res.probs
    print("INFER_OK")
    print("MASK:", m.shape, "dtype", m.dtype, "sum", int(m.sum()))
    print("PROBS:", p.shape, "dtype", p.dtype, "min/max", float(np.min(p)), float(np.max(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
