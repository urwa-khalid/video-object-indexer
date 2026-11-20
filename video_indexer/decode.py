"""Decode one mp4 (given as bytes) and choose the frames that go to the model."""
from __future__ import annotations

import io
from typing import Iterator, Optional, Tuple

import numpy as np


class VideoSampler:
    """Decode an mp4 given as bytes and yield (second, frame_bgr).

    samples_per_sec = 0 -> every decoded frame (reference mode).
    samples_per_sec = k -> each second is cut into k equal slots and the first
    frame of every slot is kept.

    Seconds are counted from the first decoded frame, exactly (integer maths on
    the stream time base): second s holds frames with 0 + s <= t < s + 1.
    The video rotation tag (display matrix) is applied to the kept frames.

    After the iteration is finished:
      n_seconds   number of seconds to report (last second holding a frame + 1)
      frames      frames decoded
      error       None, or a short message if the video broke part way
      bad_packets packets the decoder rejected (we skip those and carry on)
    """

    def __init__(self, data: bytes, samples_per_sec: int = 4, threads: int = 2,
                 skip_nonref: bool = True):
        self.data = data
        self.k = int(samples_per_sec)
        self.threads = threads
        self.skip_nonref = skip_nonref and self.k > 0
        self.n_seconds = 0
        self.frames = 0
        self.error: Optional[str] = None
        self.bad_packets = 0
        self.max_sec_seen = -1

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        import av
        import cv2

        k = self.k
        container = None
        try:
            container = av.open(io.BytesIO(self.data), buffer_size=1 << 20)
            if not container.streams.video:
                self.error = "no video stream"
                return
            stream = container.streams.video[0]
            ctx = stream.codec_context
            try:
                ctx.thread_type = "AUTO"
                ctx.thread_count = self.threads
                if self.skip_nonref and ctx.name in ("h264", "hevc"):
                    ctx.skip_frame = "NONREF"  # don't decode frames nothing else uses
            except Exception:
                pass

            pts0 = None
            max_pts = None
            rot = None
            taken = set()
            for packet in container.demux(stream):
                ppts = packet.pts if packet.pts is not None else packet.dts
                if ppts is not None and (max_pts is None or ppts > max_pts):
                    max_pts = ppts
                try:
                    frames = packet.decode()
                except av.error.FFmpegError:
                    self.bad_packets += 1
                    continue
                for frame in frames:
                    self.frames += 1
                    if frame.pts is None:
                        continue
                    tb = frame.time_base
                    if pts0 is None:
                        pts0 = frame.pts
                        num, den = tb.numerator, tb.denominator
                        rot = _rotation_code(frame, cv2)
                    ticks = frame.pts - pts0
                    if ticks < 0:
                        ticks = 0
                    scaled = ticks * num
                    sec = scaled // den
                    if sec > self.max_sec_seen:
                        self.max_sec_seen = sec
                    if k > 0:
                        slot = ((scaled - sec * den) * k) // den
                        key = (sec, slot)
                        if key in taken:
                            continue
                        taken.add(key)
                    img = frame.to_ndarray(format="bgr24")
                    if rot is not None:
                        img = cv2.rotate(img, rot)
                    yield int(sec), img

            if pts0 is None:
                self.error = "could not decode video"
                return
            end_ticks = max(0, (max_pts - pts0) if max_pts is not None else 0)
            self.n_seconds = max((end_ticks * num) // den, self.max_sec_seen) + 1
        except GeneratorExit:
            raise
        except Exception as e:  # broken container part way through
            self.error = f"{type(e).__name__}: {e}"[:150]
            # the last second we saw may be cut short -> don't report it
            self.n_seconds = max(self.max_sec_seen, 0)
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass


def _rotation_code(frame, cv2):
    """cv2.rotate code for the video's rotation tag (None = no rotation).

    frame.rotation is the counter-clockwise angle (from the display matrix) the
    frame has to be turned by to be shown upright - what ffmpeg/OpenCV autorotate does.
    """
    try:
        angle = int(round(float(getattr(frame, "rotation", 0) or 0))) % 360
    except Exception:
        return None
    return {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}.get(angle)
