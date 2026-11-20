#!/usr/bin/env python
"""Preparation step (needs internet, not timed): fetch the official yolo11l.pt next to run.py and check the install."""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

target = HERE / "yolo11l.pt"
if not target.exists():
    from ultralytics.utils.downloads import attempt_download_asset

    attempt_download_asset(str(target))
if not target.exists() or target.stat().st_size < 10_000_000:
    sys.exit("yolo11l.pt was not downloaded")

h = hashlib.sha256(target.read_bytes()).hexdigest()
print(f"yolo11l.pt ready: {target.stat().st_size / 1e6:.1f} MB, sha256 {h[:16]}...")

# load it once so a broken install shows up here and not in the timed run
from ultralytics import YOLO  # noqa: E402

m = YOLO(str(target))
assert m.names[0] == "person" and m.names[67] == "cell phone"
import torch  # noqa: E402

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# one tiny fp16 convolution per GPU: catches "no kernel image for this GPU" now, not 3 hours into a run
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        dev = f"cuda:{i}"
        conv = torch.nn.Conv2d(3, 8, 3).half().to(dev)
        conv(torch.randn(1, 3, 64, 64, device=dev).half())
        torch.cuda.synchronize(dev)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, capability {torch.cuda.get_device_capability(i)}, fp16 conv OK")
