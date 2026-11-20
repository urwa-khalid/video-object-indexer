#!/usr/bin/env python
"""Check that the fast detector path gives the same counts as the official YOLO.predict path.

    python tools/verify_detector.py --video some.mp4 [--frames 64] [--fp32]
    python tools/verify_detector.py --parquet finevideo/data/train-00000-of-01357.parquet --row 0

Run it once on the GPU machine (fp16 and --fp32) before trusting the fast path.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_indexer import Detector, NAMES, VideoSampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--parquet")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "yolo11l.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    if a.parquet:
        import pyarrow.parquet as pq

        data = pq.ParquetFile(a.parquet).read_row_group(0, columns=["mp4"]).column(0)[a.row].as_py()
    else:
        data = Path(a.video).read_bytes()

    import torch

    device = a.device if a.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    fast = Detector(a.weights, device, half=not a.fp32, mode="fast")
    off = Detector(a.weights, device, half=not a.fp32, mode="official")

    frames = [img for _, img in VideoSampler(data, samples_per_sec=2, skip_nonref=False)]
    frames = frames[: a.frames]
    cf = np.concatenate([fast.infer([fast.preprocess(f) for f in frames[i:i + 8]])
                         for i in range(0, len(frames), 8)])
    co = np.concatenate([off.infer(frames[i:i + 8]) for i in range(0, len(frames), 8)])
    same = (cf == co).all(axis=1)
    print(f"{len(frames)} frames, device {device}, {'fp32' if a.fp32 else 'fp16'}")
    print(f"frames with identical counts: {same.sum()} / {len(frames)}")
    for j, name in enumerate(NAMES):
        print(f"  {name:<11} fast total {cf[:, j].sum():4d}   official total {co[:, j].sum():4d}")
    for i in np.where(~same)[0][:10]:
        print(f"  frame {i}: fast {cf[i].tolist()} official {co[i].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
