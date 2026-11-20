"""Per-second object indexing of videos with YOLO11l."""
from .constants import COCO_IDS, NAMES
from .decode import VideoSampler
from .detector import Detector, FakeDetector
from .pipeline import process_video, summarize_second

__all__ = ["COCO_IDS", "NAMES", "VideoSampler", "Detector", "FakeDetector",
           "process_video", "summarize_second"]
