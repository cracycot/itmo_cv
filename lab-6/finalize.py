"""Finalize lab-6 after training: run val metrics, custom metrics, real_photos inference, and write results.json.

Use when training completed (or was interrupted) but the notebook didn't reach the post-training cells.
The script reads the existing best.pt and the rtsd_yolo dataset, computes everything, and saves results.json.

Usage (from repo root or lab-6/):
    python3 finalize.py
    python3 finalize.py --weights runs/rtsd_yolo11s/weights/best.pt
    python3 finalize.py --conf 0.20 --imgsz 640
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

SUPER_NAMES = ['warning', 'priority', 'prohibitory', 'mandatory', 'special', 'informational', 'service', 'additional']
SUPER_RU = ['Предупреждающие', 'Приоритета', 'Запрещающие', 'Предписывающие', 'Особых_предписаний', 'Информационные', 'Сервиса', 'Доп_информации']
NUM_CLASSES = len(SUPER_NAMES)


def load_yolo_seg_gt(label_path: Path, H: int, W: int):
    masks, classes = [], []
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


def predict_masks(model: YOLO, img_path: Path, imgsz: int, conf: float, device):
    res = model.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    H, W = res.orig_shape
    masks, classes, scores = [], [], []
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
    return masks, classes, scores


def overlay(img_rgb, masks, classes, scores=None, alpha=0.45):
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
            label = SUPER_NAMES[c] if 0 <= c < NUM_CLASSES else f'cls{c}'
            if scores is not None and k < len(scores):
                label += f' {scores[k]:.2f}'
            cv2.putText(out, label, (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(v) for v in color), 1)
    return out


def iou_pair(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def l2_pair(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt((diff * diff).sum()))


def evaluate(model, img_dir, lbl_dir, num_classes, match_iou=0.5, conf=0.25, imgsz=640, max_imgs=None, require_gt=False, device=0):
    img_files = sorted(
        glob.glob(str(Path(img_dir) / '*.jpg')) +
        glob.glob(str(Path(img_dir) / '*.jpeg')) +
        glob.glob(str(Path(img_dir) / '*.png')) +
        glob.glob(str(Path(img_dir) / '*.JPG')) +
        glob.glob(str(Path(img_dir) / '*.PNG'))
    )
    if max_imgs:
        img_files = img_files[:max_imgs]
    if require_gt:
        img_files = [ip for ip in img_files
                     if (Path(lbl_dir) / (Path(ip).stem + '.txt')).exists()
                     and (Path(lbl_dir) / (Path(ip).stem + '.txt')).stat().st_size > 0]
    tp = fp = fn = 0
    iou_per_match, l2_per_match, image_mean_iou = [], [], []
    for ip in tqdm(img_files, desc=f'eval {Path(img_dir).name}'):
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_m, gt_c = load_yolo_seg_gt(Path(lbl_dir) / (Path(ip).stem + '.txt'), H, W)
        pr_m, pr_c, pr_s = predict_masks(model, Path(ip), imgsz, conf, device)
        used_pr, used_gt = set(), set()
        per_image_ious = []
        for c in range(num_classes):
            gi = [i for i, x in enumerate(gt_c) if x == c]
            pi = [i for i, x in enumerate(pr_c) if x == c]
            if not gi or not pi:
                continue
            mat = np.zeros((len(pi), len(gi)))
            for u, p in enumerate(pi):
                for v, g in enumerate(gi):
                    mat[u, v] = iou_pair(pr_m[p], gt_m[g])
            while True:
                u, v = np.unravel_index(mat.argmax(), mat.shape)
                if mat[u, v] <= 0:
                    break
                p, g = pi[u], gi[v]
                iv = float(mat[u, v])
                if iv >= match_iou:
                    tp += 1
                    used_pr.add(p)
                    used_gt.add(g)
                    iou_per_match.append(iv)
                    l2_per_match.append(l2_pair(pr_m[p], gt_m[g]))
                    per_image_ious.append(iv)
                mat[u, :] = -1
                mat[:, v] = -1
        fp += len([i for i in range(len(pr_c)) if i not in used_pr])
        fn += len([i for i in range(len(gt_c)) if i not in used_gt])
        if len(gt_m) > 0:
            image_mean_iou.append(float(np.mean(per_image_ious)) if per_image_ious else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        'images': len(img_files),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'mean_iou_matched': round(float(np.mean(iou_per_match)), 4) if iou_per_match else 0.0,
        'mean_l2_matched': round(float(np.mean(l2_per_match)), 2) if l2_per_match else 0.0,
        'frac_iou_ge_0.50': round(float(np.mean([x >= 0.50 for x in image_mean_iou])), 4) if image_mean_iou else 0.0,
        'frac_iou_ge_0.75': round(float(np.mean([x >= 0.75 for x in image_mean_iou])), 4) if image_mean_iou else 0.0,
        'frac_iou_ge_0.90': round(float(np.mean([x >= 0.90 for x in image_mean_iou])), 4) if image_mean_iou else 0.0,
    }


def find_best_pt(lab_dir: Path) -> Path | None:
    pri = sorted(lab_dir.glob('runs/**/rtsd_yolo11s/weights/best.pt'))
    if pri:
        return pri[0]
    candidates = sorted(lab_dir.glob('runs/**/weights/best.pt'))
    return candidates[0] if candidates else None


def find_data_yaml(lab_dir: Path) -> Path | None:
    for cand in [lab_dir / 'data' / 'rtsd_yolo' / 'data.yaml']:
        if cand.exists():
            return cand
    found = list(lab_dir.glob('data/**/data.yaml'))
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default=None)
    ap.add_argument('--data-yaml', type=str, default=None)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--no-real-photos', action='store_true')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    lab = here if here.name == 'lab-6' else (here / 'lab-6' if (here / 'lab-6').exists() else here)
    results_path = lab / 'results.json'

    weights = Path(args.weights) if args.weights else find_best_pt(lab)
    if not weights or not weights.exists():
        print(f'ERROR: best.pt not found under {lab}/runs/. Pass --weights.')
        sys.exit(1)
    data_yaml = Path(args.data_yaml) if args.data_yaml else find_data_yaml(lab)
    if not data_yaml or not data_yaml.exists():
        print(f'ERROR: data.yaml not found. Pass --data-yaml.')
        sys.exit(1)

    print(f'weights : {weights}')
    print(f'data    : {data_yaml}')
    print(f'results : {results_path}')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'device  : {device}')

    model = YOLO(str(weights))

    print('\n=== Ultralytics val (mAP) ===')
    val_metrics = model.val(data=str(data_yaml), split='val', imgsz=args.imgsz, batch=args.batch, device=device, plots=False, verbose=False)
    box_map50 = round(float(val_metrics.box.map50), 4)
    box_map = round(float(val_metrics.box.map), 4)
    seg_map50 = round(float(val_metrics.seg.map50), 4)
    seg_map = round(float(val_metrics.seg.map), 4)
    print(f'  Box mAP50    : {box_map50}')
    print(f'  Box mAP50-95 : {box_map}')
    print(f'  Mask mAP50   : {seg_map50}')
    print(f'  Mask mAP50-95: {seg_map}')

    yolo_dir = data_yaml.parent
    val_imgs_dir = yolo_dir / 'images' / 'val'
    val_lbl_dir = yolo_dir / 'labels' / 'val'

    print('\n=== Custom metrics on val ===')
    val_custom = evaluate(model, val_imgs_dir, val_lbl_dir, NUM_CLASSES, conf=args.conf, imgsz=args.imgsz, device=device)
    for k, v in val_custom.items():
        print(f'  {k:>20}: {v}')

    real_metrics = None
    real_dir = lab / 'data' / 'real_photos'
    if not args.no_real_photos and real_dir.exists():
        real_imgs = sorted(
            glob.glob(str(real_dir / '*.jpg')) +
            glob.glob(str(real_dir / '*.jpeg')) +
            glob.glob(str(real_dir / '*.png')) +
            glob.glob(str(real_dir / '*.JPG')) +
            glob.glob(str(real_dir / '*.PNG'))
        )
        if real_imgs:
            print(f'\n=== Real photos inference ({len(real_imgs)}) ===')
            real_lbl_dir = real_dir / 'labels'
            real_lbl_dir.mkdir(exist_ok=True)
            real_have_gt = 0
            for ip in real_imgs:
                stem = Path(ip).stem
                existing_txt = Path(ip).with_suffix('.txt')
                out_txt = real_lbl_dir / (stem + '.txt')
                if existing_txt.exists():
                    shutil.copy2(existing_txt, out_txt)
                    if out_txt.stat().st_size > 0:
                        real_have_gt += 1
                elif not out_txt.exists():
                    out_txt.write_text('')
            pred_dir = real_dir / 'predictions'
            pred_dir.mkdir(exist_ok=True)
            summary = []
            for ip in tqdm(real_imgs, desc='real photos'):
                ip_p = Path(ip)
                img_bgr = cv2.imread(str(ip_p))
                if img_bgr is None:
                    continue
                img_rgb = img_bgr[:, :, ::-1]
                pr_m, pr_c, pr_s = predict_masks(model, ip_p, args.imgsz, args.conf, device)
                vis = overlay(img_rgb, pr_m, pr_c, pr_s)
                cv2.imwrite(str(pred_dir / ip_p.name), vis[:, :, ::-1])
                summary.append({
                    'name': ip_p.name,
                    'detections': len(pr_m),
                    'classes': [SUPER_NAMES[c] if 0 <= c < NUM_CLASSES else f'cls{c}' for c in pr_c],
                    'scores': [round(s, 3) for s in pr_s],
                })
            if real_have_gt > 0:
                real_metrics = evaluate(model, real_dir, real_lbl_dir, NUM_CLASSES, conf=args.conf, imgsz=args.imgsz, device=device, require_gt=True)
                real_metrics['per_image'] = summary
            else:
                real_metrics = {
                    'images': len(real_imgs),
                    'images_with_gt': 0,
                    'note': 'no GT annotations — only qualitative inference; metrics not computed',
                    'per_image': summary,
                }
            print(f'visualizations -> {pred_dir}')

    # Try to recover training time from results.csv
    training_seconds = None
    epochs_completed = None
    rcsv = lab / 'runs' / 'rtsd_yolo11s' / 'results.csv'
    if rcsv.exists():
        with open(rcsv) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if len(lines) > 1:
            epochs_completed = len(lines) - 1
            try:
                last = lines[-1].split(',')
                training_seconds = float(last[1])  # 'time' column
            except (ValueError, IndexError):
                pass

    results = {
        'task': 'instance_segmentation_8_road_sign_categories',
        'dataset': 'RTSD (MSU Graphics) — 8 super-classes by ПДД category',
        'super_classes': SUPER_NAMES,
        'super_classes_ru': SUPER_RU,
        'mask_strategy': 'bbox_rectangle',
        'model': 'YOLO11s-seg (pretrained COCO, fine-tuned)',
        'num_classes': NUM_CLASSES,
        'epochs_completed': epochs_completed,
        'training_seconds': round(training_seconds, 1) if training_seconds else None,
        'training_minutes': round(training_seconds / 60, 1) if training_seconds else None,
        'best_weights': str(weights),
        'data_yaml': str(data_yaml),
        'ultralytics_metrics': {
            'box_mAP50': box_map50,
            'box_mAP50_95': box_map,
            'mask_mAP50': seg_map50,
            'mask_mAP50_95': seg_map,
        },
        'custom_metrics_val': val_custom,
        'custom_metrics_real_photos': real_metrics,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n--- saved {results_path} ---')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
