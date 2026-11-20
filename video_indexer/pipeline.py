"""One video: bytes of an mp4 -> one (second, objects, people) row per second."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from .constants import NAMES
from .decode import VideoSampler


def summarize_second(counts: np.ndarray) -> Tuple[List[str], int]:
    """counts: (n_frames, 5) ints, column 0 = number of people in that frame.

    An object is present if it is found in at least half of the frames.
    People = median of the per-frame counts, rounded down when halfway.
    """
    n = len(counts)
    if n == 0:
        return [], 0
    present = (counts > 0).sum(axis=0)
    objects = [NAMES[j] for j in range(len(NAMES)) if present[j] * 2 >= n]
    s = np.sort(counts[:, 0])
    if n % 2:
        people = int(s[n // 2])
    else:
        people = (int(s[n // 2 - 1]) + int(s[n // 2])) // 2
    return objects, people


def process_video(det, data: bytes, samples_per_sec: int = 4, batch: int = 16,
                  threads: int = 2, skip_nonref: bool = True,
                  stats: Optional[dict] = None):
    """Returns (rows, error). rows = [(second, objects, people), ...], one per second."""
    st = stats if stats is not None else defaultdict(float)
    sampler = VideoSampler(data, samples_per_sec, threads, skip_nonref)
    per_sec = defaultdict(list)
    buf, buf_sec = [], []
    pc = time.perf_counter

    def flush():
        if not buf:
            return
        t = pc()
        counts = det.infer(buf)  # ends with .cpu(), so this time includes the GPU work
        st["infer"] += pc() - t
        st["frames_inferred"] += len(buf)
        for s, c in zip(buf_sec, counts):
            per_sec[s].append(c)
        buf.clear()
        buf_sec.clear()

    it = iter(sampler)
    while True:
        t = pc()
        try:
            sec, img = next(it)
        except StopIteration:
            st["decode"] += pc() - t
            break
        st["decode"] += pc() - t
        t = pc()
        item = det.preprocess(img)
        st["preprocess"] += pc() - t
        if buf and item.shape != buf[0].shape:
            flush()
        buf.append(item)
        buf_sec.append(sec)
        if len(buf) >= batch:
            flush()
    flush()
    st["frames_decoded"] += sampler.frames

    error = sampler.error
    if error and sampler.n_seconds == 0 and not per_sec:
        return [], error
    rows = []
    for s in range(sampler.n_seconds):
        c = per_sec.get(s)
        if c:
            objects, people = summarize_second(np.stack(c))
        else:
            objects, people = [], 0
        rows.append((s, objects, people))
    return rows, error
