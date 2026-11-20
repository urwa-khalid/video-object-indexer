"""Logic tests that need no GPU and no model:  python -m pytest -q tests"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import expected_second, make_video, rotate_with_ffmpeg, write_parquet  # noqa: E402
from video_indexer import FakeDetector, VideoSampler, process_video, summarize_second  # noqa: E402


# ---------------------------------------------------------------- the per-second rule
def test_summarize_second_rules():
    c = lambda rows: np.array(rows)  # noqa: E731
    # "at least half of the frames" -> 2 of 4 is enough, 1 of 4 is not
    assert summarize_second(c([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]))[0] == ["person", "car"]
    assert summarize_second(c([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]))[0] == ["person"]
    # median, rounded down when halfway: [0, 1] -> 0 ; [1, 2] -> 1 ; [0, 2] -> 1
    assert summarize_second(c([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]))[1] == 0
    assert summarize_second(c([[1, 0, 0, 0, 0], [2, 0, 0, 0, 0]]))[1] == 1
    assert summarize_second(c([[0, 0, 0, 0, 0], [2, 0, 0, 0, 0]]))[1] == 1
    assert summarize_second(c([[3, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0]]))[1] == 2
    assert summarize_second(np.zeros((0, 5), int)) == ([], 0)


# ---------------------------------------------------------------- seconds and frame choice
@pytest.mark.parametrize("n_frames,fps,want_seconds", [
    (175, 25, 7),    # last frame at 6.96 s -> second 6 -> 7 lines
    (176, 25, 8),    # last frame at exactly 7.00 s -> second 7 -> 8 lines
    (300, 30000 / 1001, 10),  # 29.97 fps: last frame at 9.98 s
    (25, 25, 1),     # exactly one second
    (1, 25, 1),      # a single frame
])
def test_number_of_seconds(n_frames, fps, want_seconds):
    data = make_video(n_frames=n_frames, fps=fps)
    for k in (0, 4):
        s = VideoSampler(data, samples_per_sec=k)
        list(s)
        assert s.error is None
        assert s.n_seconds == want_seconds, (k, s.n_seconds)


@pytest.mark.parametrize("bframes", [0, 2])
@pytest.mark.parametrize("k", [0, 1, 3, 4, 8])
def test_rows_match_expected_every_k(bframes, k):
    data = make_video(n_frames=175, fps=25, bframes=bframes)
    rows, error = process_video(FakeDetector(), data, samples_per_sec=k, batch=5)
    assert error is None
    assert [r[0] for r in rows] == list(range(7))
    for sec, objects, people in rows:
        assert (objects, people) == expected_second(sec), (sec, k)


def test_sampling_takes_k_frames_per_second():
    data = make_video(n_frames=100, fps=25)
    per_sec = {}
    for sec, _ in VideoSampler(data, samples_per_sec=4, skip_nonref=False):
        per_sec[sec] = per_sec.get(sec, 0) + 1
    assert per_sec == {0: 4, 1: 4, 2: 4, 3: 4}
    n_all = sum(1 for _ in VideoSampler(data, samples_per_sec=0))
    assert n_all == 100


def test_skipping_nonref_frames_does_not_change_seconds():
    data = make_video(n_frames=176, fps=25, bframes=3)
    a = VideoSampler(data, 4, skip_nonref=True)
    b = VideoSampler(data, 4, skip_nonref=False)
    list(a), list(b)
    assert a.n_seconds == b.n_seconds == 8


# ---------------------------------------------------------------- rotation
@pytest.mark.parametrize("degrees,swap", [(90, True), (-90, True), (180, False)])
def test_rotation_tag_is_applied(degrees, swap):
    data = rotate_with_ffmpeg(make_video(n_frames=30, size=(160, 96)), degrees)
    if data is None:
        pytest.skip("ffmpeg command line tool not available")
    (_, img), *_ = list(VideoSampler(data, samples_per_sec=1))
    assert img.shape[:2] == ((160, 96) if swap else (96, 160))


# ---------------------------------------------------------------- broken input
def test_garbage_and_truncated_videos_give_errors_not_crashes():
    good = make_video(n_frames=75)
    for bad in (b"this is not a video", b"", good[: len(good) // 3]):
        rows, error = process_video(FakeDetector(), bad, samples_per_sec=4)
        assert error, "broken input must produce an error message"


# ---------------------------------------------------------------- the whole program
def run_cli(tmp_path, videos, extra=()):
    folder = tmp_path / "in"
    folder.mkdir()
    write_parquet(folder / "train-00000-of-01357.parquet", videos[:3])
    write_parquet(folder / "train-00001-of-01357.parquet", videos[3:])
    out = tmp_path / "results.jsonl"
    cmd = [sys.executable, str(ROOT / "run.py"), "--input", str(folder), "--output", str(out),
           "--detector", "fake", "--workers", "2", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return r, [json.loads(line) for line in out.read_text().splitlines()]


def test_cli_end_to_end(tmp_path):
    v = make_video(n_frames=175)
    videos = [v, b"not a video", make_video(n_frames=50), None, v]
    r, lines = run_cli(tmp_path, videos)
    assert r.returncode == 0, r.stderr
    by_video = {}
    for d in lines:
        by_video.setdefault(d["video"], []).append(d)
    assert set(by_video) == {"train-00000-of-01357.parquet:0", "train-00000-of-01357.parquet:1",
                             "train-00000-of-01357.parquet:2", "train-00001-of-01357.parquet:0",
                             "train-00001-of-01357.parquet:1"}
    assert len(by_video["train-00000-of-01357.parquet:0"]) == 7
    assert len(by_video["train-00000-of-01357.parquet:2"]) == 2
    for name in ("train-00000-of-01357.parquet:1", "train-00001-of-01357.parquet:0"):
        assert len(by_video[name]) == 1 and "error" in by_video[name][0]
    for d in by_video["train-00001-of-01357.parquet:1"]:
        objects, people = expected_second(d["second"])
        assert d["objects"] == objects and d["people"] == people
        assert set(d) == {"video", "second", "objects", "people"}


def test_cli_survives_a_worker_crash(tmp_path):
    v = make_video(n_frames=50)
    r, lines = run_cli(tmp_path, [v, b"CRASHME" + b"\0" * 50, v, v, v])
    assert r.returncode == 0, r.stderr
    errors = [d for d in lines if "error" in d]
    assert [d["video"] for d in errors] == ["train-00000-of-01357.parquet:1"]
    ok_videos = {d["video"] for d in lines if "error" not in d}
    assert len(ok_videos) == 4


def test_cli_resume_gives_same_result_as_one_go(tmp_path):
    v = make_video(n_frames=75)
    videos = [v, make_video(n_frames=50), v, v, make_video(n_frames=100)]
    folder = tmp_path / "in"
    folder.mkdir()
    write_parquet(folder / "train-00000-of-01357.parquet", videos[:3])
    write_parquet(folder / "train-00001-of-01357.parquet", videos[3:])
    base = [sys.executable, str(ROOT / "run.py"), "--input", str(folder), "--detector", "fake", "--workers", "2"]

    full = tmp_path / "full.jsonl"
    assert subprocess.run(base + ["--output", str(full)], capture_output=True).returncode == 0

    part = tmp_path / "part.jsonl"
    assert subprocess.run(base + ["--output", str(part), "--max-videos", "3"], capture_output=True).returncode == 0
    with open(part, "a") as f:  # simulate being killed in the middle of writing a line
        f.write('{"video": "train-00001-of-01357.parquet:0", "sec')
    r = subprocess.run(base + ["--output", str(part), "--resume"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    load = lambda p: sorted(p.read_text().splitlines())  # noqa: E731
    assert load(part) == load(full)
