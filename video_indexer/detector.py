"""YOLO11l wrapper: a batch of frames in, per-frame counts of the five classes out."""
from __future__ import annotations

import numpy as np

from .constants import COCO_IDS, NAMES


class FakeDetector:
    """Test-only detector: counts come from the picture brightness (no model, no GPU)."""

    def preprocess(self, img):
        return img

    def infer(self, imgs):
        out = np.zeros((len(imgs), len(NAMES)), dtype=np.int32)
        for i, im in enumerate(imgs):
            m = float(im.mean())
            out[i, 0] = int(m // 40) % 4  # people 0..3
            out[i, 1] = 1 if m > 120 else 0  # car
            out[i, 3] = 1 if 60 < m < 160 else 0  # laptop
        return out


class Detector:
    """YOLO11l -> per-frame counts for the five classes (column 0 = people).

    mode "fast":     run the network ourselves (Ultralytics letterbox + Ultralytics NMS),
                     skipping the Results objects of model.predict().
    mode "official": call YOLO.predict() - slowest, but the exact "official" path.
    """

    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True,
                 mode: str = "fast", imgsz: int = 640, conf: float = 0.4,
                 iou: float = 0.7):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = torch.device(device)
        self.half = bool(half) and self.device.type == "cuda"
        self.mode = mode
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.yolo = YOLO(weights)
        names = self.yolo.names
        for cid, name in zip(COCO_IDS, NAMES):
            assert names[cid] == name, f"class {cid} is {names[cid]!r}, expected {name!r}"
        if mode == "fast":
            from ultralytics.data.augment import LetterBox

            net = self.yolo.model.fuse().to(self.device).eval()
            net = net.half() if self.half else net.float()
            self.net = net
            stride = int(max(net.stride)) if hasattr(net, "stride") else 32
            # auto=True: pad only up to a multiple of the stride (what model.predict does
            # when all the frames of a batch have the same shape)
            self.letterbox = LetterBox((imgsz, imgsz), auto=True, stride=stride)
            try:  # NMS moved between Ultralytics versions
                from ultralytics.utils.nms import non_max_suppression
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression

            self.nms = non_max_suppression

    # -- CPU side (runs in the worker, parallel to the GPU of other workers) --
    def preprocess(self, img_bgr: np.ndarray):
        if self.mode == "fast":
            return self.letterbox(image=img_bgr)
        return img_bgr

    # -- GPU side --
    def infer(self, items) -> np.ndarray:
        if self.mode == "official":
            return self._infer_official(items)
        try:
            return self._infer_fast(items)
        except self.torch.OutOfMemoryError:
            if len(items) == 1:
                raise
            self.torch.cuda.empty_cache()
            h = len(items) // 2
            return np.concatenate([self.infer(items[:h]), self.infer(items[h:])])

    def _infer_fast(self, items) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.stack(items)).to(self.device, non_blocking=True)  # B,H,W,3 uint8 BGR
        x = x.flip(-1).permute(0, 3, 1, 2).contiguous()  # -> RGB, B,3,H,W
        x = x.half() if self.half else x.float()
        x = x / 255.0
        with torch.inference_mode():
            preds = self.net(x)
            dets = self.nms(
                preds, self.conf, self.iou, classes=COCO_IDS, agnostic=False,
                max_det=300, max_time_img=60.0)  # never let NMS give up early
            n = len(dets)
            img_idx = torch.cat([torch.full((len(d),), i, device=d.device, dtype=torch.long)
                                 for i, d in enumerate(dets)])
            cls = torch.cat([d[:, 5] for d in dets]).long()
            table = torch.bincount(img_idx * 80 + cls, minlength=n * 80).view(n, 80)
            counts = table[:, COCO_IDS].cpu().numpy().astype(np.int32)
        return counts

    def _infer_official(self, frames) -> np.ndarray:
        res = self.yolo.predict(list(frames), imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                half=self.half, device=str(self.device), verbose=False)
        out = np.zeros((len(frames), len(NAMES)), dtype=np.int32)
        for i, r in enumerate(res):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for j, cid in enumerate(COCO_IDS):
                out[i, j] = int((cls == cid).sum())
        return out
