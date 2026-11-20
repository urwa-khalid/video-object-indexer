"""Tiny synthetic videos + parquet files for the tests (no real data, no GPU needed).

Every second of a test video is a flat grey whose level depends on the second:
LEVELS[second % 5]. The FakeDetector in video_indexer/detector.py turns brightness into counts,
so the expected output of every second is known in advance.
"""
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

LEVELS = [20, 70, 110, 150, 190]


def expected_second(sec):
    """(objects, people) the FakeDetector produces for second `sec` of a test video."""
    level = LEVELS[sec % 5]
    people = int(level // 40) % 4
    objects = []
    if people > 0:
        objects.append("person")
    if level > 120:
        objects.append("car")
    if 60 < level < 160:
        objects.append("laptop")
    return objects, people


def make_video(n_frames=175, fps=25, size=(160, 96), gop=25, bframes=0):
    """mp4 bytes. Frame i is grey LEVELS[int(i / fps) % 5]."""
    import av
    from fractions import Fraction

    w, h = size
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(100000))
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"g": str(gop), "bf": str(bframes), "crf": "10"}
        for i in range(n_frames):
            level = LEVELS[int(i / float(fps)) % 5]
            f = av.VideoFrame.from_ndarray(np.full((h, w, 3), level, np.uint8), format="rgb24")
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def rotate_with_ffmpeg(mp4: bytes, degrees: int):
    """Same video with a display-matrix rotation tag (needs the ffmpeg command line tool)."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.mp4", Path(d) / "b.mp4"
        src.write_bytes(mp4)
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation",
                            str(degrees), "-i", str(src), "-c", "copy", str(dst)], capture_output=True)
        return dst.read_bytes() if r.returncode == 0 else None


def write_parquet(path, videos, row_group_size=2):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"mp4": pa.array(videos, type=pa.binary()),
                      "json": pa.array([json.dumps({"i": i}) for i in range(len(videos))])})
    pq.write_table(table, path, row_group_size=row_group_size)
