"""Resume YOLO11s-seg training from last.pt for lab-6.

Continues from where training stopped (e.g. SSH disconnect killed the process).
Preserves all original hyperparameters via Ultralytics' `resume=True`:
    - epochs=80, batch=16, imgsz=640, patience=20, save_period=3, etc.

Use inside tmux to survive SSH disconnects:
    tmux new -s train
    source /mnt/data/itmo-cv/.venv/bin/activate
    cd /mnt/data/itmo-cv
    python3 lab-6/resume_train.py
    # Ctrl+B then D to detach, `tmux attach -t train` to come back

After it finishes:
    python3 lab-6/finalize.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def find_last_pt(lab_dir: Path) -> Path | None:
    pri = sorted(lab_dir.glob('runs/**/rtsd_yolo11s/weights/last.pt'))
    if pri:
        return pri[0]
    found = sorted(lab_dir.glob('runs/**/weights/last.pt'))
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default=None,
                    help='path to last.pt; auto-detected from runs/rtsd_yolo11s/ by default')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    lab = here if here.name == 'lab-6' else (here / 'lab-6' if (here / 'lab-6').exists() else here)

    weights = Path(args.weights) if args.weights else find_last_pt(lab)
    if not weights or not weights.exists():
        print(f'ERROR: last.pt not found under {lab}/runs/. Pass --weights.')
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'weights : {weights}')
    print(f'device  : {device}')
    print(f'GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"}')

    model = YOLO(str(weights))
    print('\nResuming training (Ultralytics will read all original args from runs/.../args.yaml)...')

    t_start = time.time()
    try:
        model.train(resume=True)
    except KeyboardInterrupt:
        print('\nInterrupted by user — last.pt is up-to-date, can resume again.')
        sys.exit(130)
    elapsed = time.time() - t_start
    print(f'\nresume took {elapsed/60:.1f} minutes')


if __name__ == '__main__':
    main()
