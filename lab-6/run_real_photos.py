"""Run inference of fine-tuned YOLO11s-seg on data/real_photos/ and update results.json.

Usage (from repo root or lab-6/):
    python3 run_real_photos.py
    python3 run_real_photos.py --conf 0.20 --imgsz 768
    python3 run_real_photos.py --weights /path/to/best.pt

Behavior:
    * loads best.pt from runs/segment/signs_yolo11s/weights/best.pt (or signs_yolo11s)
    * iterates over data/real_photos/*.{jpg,jpeg,png,JPG,PNG}
    * saves visualizations to data/real_photos/predictions/<name>
    * if <name>.txt (YOLO seg) or <name>.jpg_coco.json (Mask R-CNN format) exists:
        computes IoU, Precision, Recall, L2, %IoU>=0.5/0.75/0.9 on the annotated subset
    * patches results.json with custom_metrics_real_photos = {...}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

CLASS_NAMES = ['warning', 'priority', 'prohibitory', 'mandatory', 'special', 'informational', 'service', 'additional']
CLASS_NAMES_RU = ['Предупреждающие', 'Приоритета', 'Запрещающие', 'Предписывающие', 'Особых_предписаний', 'Информационные', 'Сервиса', 'Доп_информации']
NUM_CLASSES = len(CLASS_NAMES)
# COCO-pseudo-label remap (used only when reading legacy *_coco.json annotations from old Kaggle dataset)
CLASS_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5, 10: 6, 13: 7}


def coco_json_to_yolo_txt(json_path: Path, img_path: Path, out_label_path: Path) -> tuple[bool, int]:
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0
    H, W = img.shape[:2]
    with open(json_path) as f:
        d = json.load(f)
    masks = np.asarray(d['masks'], dtype=np.uint8) if d.get('masks') else None
    rois = d.get('rois', [])
    cids = d.get('class_ids', [])
    lines: list[str] = []
    if masks is not None and masks.size > 0:
        for i, (cid, roi) in enumerate(zip(cids, rois)):
            if cid not in CLASS_REMAP:
                continue
            y1, x1, y2, x2 = (int(v) for v in roi)
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(H, y2), min(W, x2)
            if y2 <= y1 + 1 or x2 <= x1 + 1:
                continue
            m56 = masks[:, :, i]
            m_full = cv2.resize(m56, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((H, W), dtype=np.uint8)
            canvas[y1:y2, x1:x2] = m_full
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 8:
                continue
            eps = max(0.5, 0.0025 * cv2.arcLength(cnt, True))
            poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(poly) < 3:
                continue
            coords: list[float] = []
            for x, y in poly:
                coords.append(min(max(float(x) / W, 0.0), 1.0))
                coords.append(min(max(float(y) / H, 0.0), 1.0))
            lines.append(f'{CLASS_REMAP[cid]} ' + ' '.join(f'{c:.6f}' for c in coords))
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, 'w') as f:
        f.write('\n'.join(lines))
    return True, len(lines)


def load_yolo_seg_gt(label_path: Path, H: int, W: int) -> tuple[list[np.ndarray], list[int]]:
    masks: list[np.ndarray] = []
    classes: list[int] = []
    if not label_path.exists():
        return masks, classes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            coords = np.asarray([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= W
            coords[:, 1] *= H
            poly = coords.astype(np.int32)
            m = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(m, [poly], 1)
            masks.append(m.astype(bool))
            classes.append(cls)
    return masks, classes


def predict_masks(model: YOLO, img_path: Path, imgsz: int, conf: float, device) -> tuple[list[np.ndarray], list[int], list[float], tuple[int, int]]:
    res = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    H, W = res.orig_shape
    masks: list[np.ndarray] = []
    classes: list[int] = []
    scores: list[float] = []
    if res.masks is not None and len(res.masks) > 0:
        m_arr = res.masks.data.cpu().numpy()
        cls_arr = res.boxes.cls.cpu().numpy().astype(int)
        scr_arr = res.boxes.conf.cpu().numpy()
        for k in range(m_arr.shape[0]):
            m = m_arr[k]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(m.astype(bool))
            classes.append(int(cls_arr[k]))
            scores.append(float(scr_arr[k]))
    return masks, classes, scores, (H, W)


def overlay(img_rgb: np.ndarray, masks: list[np.ndarray], classes: list[int], scores: list[float] | None = None, alpha: float = 0.45) -> np.ndarray:
    rng = np.random.default_rng(123)
    palette = rng.integers(60, 255, size=(NUM_CLASSES, 3), dtype=np.int32)
    out = img_rgb.copy()
    for k, (m, c) in enumerate(zip(masks, classes)):
        color = palette[c % NUM_CLASSES]
        layer = out.copy()
        layer[m] = color
        out = cv2.addWeighted(out, 1 - alpha, layer, alpha, 0)
        ys, xs = np.where(m)
        if len(xs):
            x0, y0 = int(xs.min()), int(ys.min())
            label = CLASS_NAMES[c] if 0 <= c < NUM_CLASSES else f'cls{c}'
            if scores is not None and k < len(scores):
                label += f' {scores[k]:.2f}'
            cv2.putText(out, label, (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(v) for v in color), 1)
    return out


def iou_pair(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def l2_pair(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt((diff * diff).sum()))


def find_best_pt(lab_dir: Path) -> Path | None:
    """Find best.pt — prefer rtsd_yolo11s, then any other run."""
    pri = sorted(lab_dir.glob('runs/**/rtsd_yolo11s/weights/best.pt'))
    if pri:
        return pri[0]
    candidates = sorted(lab_dir.glob('runs/**/weights/best.pt'))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default=None, help='path to best.pt; auto-detected by default')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--match-iou', type=float, default=0.5)
    ap.add_argument('--photos-dir', type=str, default=None, help='override data/real_photos/')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    lab = here if here.name == 'lab-6' else (here / 'lab-6' if (here / 'lab-6').exists() else here)
    real_dir = Path(args.photos_dir) if args.photos_dir else (lab / 'data' / 'real_photos')
    results_path = lab / 'results.json'

    weights = Path(args.weights) if args.weights else find_best_pt(lab)
    if weights is None or not weights.exists():
        print(f'ERROR: best.pt not found. Looked under {lab}/runs/. Run training first or pass --weights.')
        sys.exit(1)
    print(f'weights : {weights}')
    print(f'photos  : {real_dir}')
    print(f'results : {results_path}')

    real_dir.mkdir(parents=True, exist_ok=True)
    real_imgs = sorted(
        glob.glob(str(real_dir / '*.jpg')) +
        glob.glob(str(real_dir / '*.jpeg')) +
        glob.glob(str(real_dir / '*.png')) +
        glob.glob(str(real_dir / '*.JPG')) +
        glob.glob(str(real_dir / '*.JPEG')) +
        glob.glob(str(real_dir / '*.PNG'))
    )
    if not real_imgs:
        print(f'no photos in {real_dir} — drop 10 jpg/png there and re-run.')
        sys.exit(0)
    print(f'found {len(real_imgs)} photos')

    real_lbl_dir = real_dir / 'labels'
    real_lbl_dir.mkdir(exist_ok=True)
    annotated = 0
    for ip in real_imgs:
        ip_p = Path(ip)
        out_txt = real_lbl_dir / (ip_p.stem + '.txt')
        existing_txt = ip_p.with_suffix('.txt')
        existing_json = Path(str(ip_p) + '_coco.json')
        if existing_txt.exists():
            shutil.copy2(existing_txt, out_txt)
            if out_txt.stat().st_size > 0:
                annotated += 1
        elif existing_json.exists():
            ok, _ = coco_json_to_yolo_txt(existing_json, ip_p, out_txt)
            if ok and out_txt.stat().st_size > 0:
                annotated += 1
        elif not out_txt.exists():
            out_txt.write_text('')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'device  : {device}')
    model = YOLO(str(weights))

    pred_dir = real_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)

    tp = fp = fn = 0
    iou_per_match: list[float] = []
    l2_per_match: list[float] = []
    image_mean_iou: list[float] = []
    summary: list[dict] = []

    t0 = time.time()
    for ip in tqdm(real_imgs, desc='inference'):
        ip_p = Path(ip)
        img_bgr = cv2.imread(str(ip_p))
        if img_bgr is None:
            print(f'skip unreadable: {ip}')
            continue
        H, W = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1]
        pr_m, pr_c, pr_s, _ = predict_masks(model, ip_p, args.imgsz, args.conf, device)

        vis_rgb = overlay(img_rgb, pr_m, pr_c, pr_s)
        cv2.imwrite(str(pred_dir / ip_p.name), vis_rgb[:, :, ::-1])

        lbl = real_lbl_dir / (ip_p.stem + '.txt')
        gt_m, gt_c = load_yolo_seg_gt(lbl, H, W)
        has_gt = len(gt_m) > 0

        per_image_ious: list[float] = []
        if has_gt:
            used_pr: set[int] = set()
            used_gt: set[int] = set()
            for c in range(NUM_CLASSES):
                gt_idx = [i for i, x in enumerate(gt_c) if x == c]
                pr_idx = [i for i, x in enumerate(pr_c) if x == c]
                if not gt_idx or not pr_idx:
                    continue
                mat = np.zeros((len(pr_idx), len(gt_idx)))
                for pi, p in enumerate(pr_idx):
                    for gi, g in enumerate(gt_idx):
                        mat[pi, gi] = iou_pair(pr_m[p], gt_m[g])
                while True:
                    pi, gi = np.unravel_index(mat.argmax(), mat.shape)
                    if mat[pi, gi] <= 0:
                        break
                    p, g = pr_idx[pi], gt_idx[gi]
                    iou_val = float(mat[pi, gi])
                    if iou_val >= args.match_iou:
                        tp += 1
                        used_pr.add(p)
                        used_gt.add(g)
                        iou_per_match.append(iou_val)
                        l2_per_match.append(l2_pair(pr_m[p], gt_m[g]))
                        per_image_ious.append(iou_val)
                    mat[pi, :] = -1
                    mat[:, gi] = -1
            fp += len([i for i in range(len(pr_c)) if i not in used_pr])
            fn += len([i for i in range(len(gt_c)) if i not in used_gt])
            image_mean_iou.append(float(np.mean(per_image_ious)) if per_image_ious else 0.0)

        summary.append({
            'name': ip_p.name,
            'detections': len(pr_m),
            'classes': [CLASS_NAMES[c] if 0 <= c < NUM_CLASSES else f'cls{c}' for c in pr_c],
            'scores': [round(s, 3) for s in pr_s],
            'has_gt': has_gt,
            'image_mean_iou': round(float(np.mean(per_image_ious)), 4) if per_image_ious else None,
        })

    elapsed = time.time() - t0

    real_metrics: dict | None
    if annotated > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        real_metrics = {
            'images': len(real_imgs),
            'images_with_gt': annotated,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'mean_iou_matched': round(float(np.mean(iou_per_match)), 4) if iou_per_match else 0.0,
            'mean_l2_matched': round(float(np.mean(l2_per_match)), 2) if l2_per_match else 0.0,
            'frac_iou_ge_0.50': round(float(np.mean([x >= 0.50 for x in image_mean_iou])), 4) if image_mean_iou else 0.0,
            'frac_iou_ge_0.75': round(float(np.mean([x >= 0.75 for x in image_mean_iou])), 4) if image_mean_iou else 0.0,
            'frac_iou_ge_0.90': round(float(np.mean([x >= 0.90 for x in image_mean_iou])), 4) if image_mean_iou else 0.0,
            'inference_seconds': round(elapsed, 2),
            'per_image': summary,
        }
    else:
        real_metrics = {
            'images': len(real_imgs),
            'images_with_gt': 0,
            'note': 'no GT annotations — only qualitative inference; metrics not computed',
            'inference_seconds': round(elapsed, 2),
            'per_image': summary,
        }

    print('\n=== real_photos metrics ===')
    for k, v in real_metrics.items():
        if k == 'per_image':
            continue
        print(f'  {k:>22}: {v}')
    print(f'\nvisualizations -> {pred_dir}')

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        results['custom_metrics_real_photos'] = real_metrics
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'updated {results_path}')
    else:
        print(f'no {results_path}, not patching (run the notebook first)')


if __name__ == '__main__':
    main()
