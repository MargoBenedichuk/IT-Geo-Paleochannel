# tools/test_sam_onnx_manual.py
# Manual interactive test for libs/ML/inference_sam_onnx.py (SamOnnxModel)

import os
import glob
import cv2
import numpy as np

# IMPORTANT: run from project root, so that "libs" is importable:
#   cd C:\Users\Ilya\river_dataset_1
#   python .\tools\test_sam_onnx_manual.py
from inference_sam_onnx import SamOnnxModel

# ================= PATHS =================
PROJECT_DIR = r"C:\Users\Ilya\river_dataset_1"
ONNX_PATH   = r'C:\Users\Ilya\river_dataset_1\sam_weights_2\SAM_river_unlimited.onnx'  # или SAM_river_unlimited.onnx
IMG_DIR     = r"C:\Users\Ilya\river_dataset_1\data\river_jpg_2"
# OUT_DIR     = os.path.join(PROJECT_DIR, "unused_data_fixed_out")
OUT_DIR     = r"C:\Users\Ilya\river_dataset_1\data\rivers_masks_png_2"

# ================= UI =================
THRESH = 0.5
REMOVE_RADIUS = 30

def list_images(img_dir: str):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    items = []
    for e in exts:
        items += glob.glob(os.path.join(img_dir, "**", e), recursive=True)
        items += glob.glob(os.path.join(img_dir, "**", e.upper()), recursive=True)
    return sorted(set(items))

def remove_nearest(click_events, x, y, max_dist=REMOVE_RADIUS):
    if not click_events:
        return
    pts = np.array([[ex, ey] for ex, ey, _ in click_events], dtype=np.float32)
    d = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
    i = int(np.argmin(d))
    if float(d[i]) <= max_dist:
        click_events.pop(i)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    items = list_images(IMG_DIR)
    if not items:
        raise SystemExit(f"No images in {IMG_DIR}")

    model = SamOnnxModel(ONNX_PATH, thresh=THRESH, prefer_cuda=True)
    print("ORT providers:", model.session.get_providers())
    print("Fixed N points (if any):", model.fixed_n_points)
    print("Has input_labels:", model.in_labels is not None)

    win = "SAM ONNX MANUAL | LMB:+ RMB:- | Backspace=del | s=save n=next r=reset q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for p in items:
        rel = os.path.relpath(p, IMG_DIR)
        safe = os.path.splitext(rel)[0].replace("\\", "__").replace("/", "__")
        out_path = os.path.join(OUT_DIR, safe + ".png")

        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            print("[SKIP]", p)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        click_events = []  # [(x,y,label)] label 1 pos, 0 neg
        mask01 = None
        last_mouse = [rgb.shape[1] // 2, rgb.shape[0] // 2]

        def run():
            nonlocal mask01
            if not click_events:
                mask01 = None
                return
            points = [(x, y) for (x, y, _) in click_events]
            labels = [int(lab) for (_, _, lab) in click_events]
            res = model.infer(
                rgb,
                points=points,
                labels=labels,
                return_probs=False,
                keep_pos_components=True,
                min_area=0,
            )
            mask01 = res.mask  # uint8 {0,1}

        def on_mouse(event, x, y, flags, param):
            last_mouse[0], last_mouse[1] = x, y
            if event == cv2.EVENT_LBUTTONDOWN:
                click_events.append((x, y, 1))
                run()
            elif event == cv2.EVENT_RBUTTONDOWN:
                click_events.append((x, y, 0))
                run()

        cv2.setMouseCallback(win, on_mouse)

        while True:
            vis = bgr.copy()

            # overlay
            if mask01 is not None and mask01.any():
                overlay = vis.copy()
                overlay[mask01.astype(bool)] = (255, 0, 255)
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # points
            for x, y, lab in click_events:
                c = (0, 255, 0) if int(lab) == 1 else (0, 0, 255)
                cv2.circle(vis, (int(x), int(y)), 4, c, -1)

            pos_n = sum(1 for _, _, lab in click_events if int(lab) == 1)
            neg_n = len(click_events) - pos_n
            cv2.putText(
                vis,
                f"{rel} | +{pos_n} -{neg_n} | model_fixedN={model.fixed_n_points}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, vis)
            k = cv2.waitKey(20) & 0xFF

            if k == ord("q"):
                cv2.destroyAllWindows()
                return

            if k == ord("r"):
                click_events.clear()
                mask01 = None

            if k == ord("n"):
                break

            if k == ord("s"):
                if mask01 is not None:
                    cv2.imwrite(out_path, (mask01 * 255).astype(np.uint8))
                    print("saved:", out_path)
                break

            if k == 8:  # backspace
                remove_nearest(click_events, last_mouse[0], last_mouse[1])
                run()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
