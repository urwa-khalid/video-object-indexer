#!/usr/bin/env python
"""Index videos: for every second of every video, which of 5 objects appear + how many people.

    python run.py --input <folder of .parquet files> --output results.jsonl

Layout: the main process reads the Parquet rows (one mp4 per row) and hands the
bytes to N worker processes. Every worker decodes a video on the CPU, runs
YOLO11l on a few frames per second on the (shared) GPU, applies the per-second
rule from the brief and sends back ready-made JSON lines. The main process
writes them. A worker that crashes costs one error line, not the whole run.
"""
import os

# one CPU thread per library inside each worker; parallelism comes from the worker processes
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

import argparse
import json
import multiprocessing as mp
import queue as pyqueue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with .parquet files (searched recursively)")
    ap.add_argument("--output", required=True, help="results .jsonl file")
    ap.add_argument("--samples-per-sec", "-k", type=int, default=4,
                    help="frames per second sent to the model (0 = every frame = reference mode)")
    ap.add_argument("--workers", type=int, default=max(2, min(16, (os.cpu_count() or 4) // 2)),
                    help="worker processes (each decodes and owns a copy of the model)")
    ap.add_argument("--decode-threads", type=int, default=2, help="ffmpeg threads per worker")
    ap.add_argument("--batch", type=int, default=16, help="frames per model call")
    ap.add_argument("--detector", choices=["fast", "official", "fake"], default="fast",
                    help="fast = own forward+NMS, official = YOLO.predict, fake = tests only")
    ap.add_argument("--fp32", action="store_true", help="float32 instead of float16")
    ap.add_argument("--device", default="auto",
                    help="auto = spread workers over all visible GPUs (e.g. Kaggle T4 x2), or cpu, cuda:0 ...")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep --output, skip the videos already in it")
    ap.add_argument("--weights", default=str(HERE / "yolo11l.pt"))
    ap.add_argument("--no-skip-nonref", action="store_true",
                    help="decode every frame even in sampling mode (slower, for checking)")
    ap.add_argument("--profile", action="store_true", help="print where the time goes")
    ap.add_argument("--max-files", type=int, default=0, help="debug: only the first N parquet files")
    ap.add_argument("--max-videos", type=int, default=0, help="debug: only the first N videos")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Worker process
# --------------------------------------------------------------------------- #
def build_detector(cfg, wid=0):
    from video_indexer import Detector, FakeDetector

    if cfg["detector"] == "fake":
        return FakeDetector()
    import torch

    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    device = cfg["device"]
    if device == "auto":
        n_gpu = torch.cuda.device_count()
        device = f"cuda:{wid % n_gpu}" if n_gpu else "cpu"  # worker i -> GPU i mod n
        if device == "cpu":
            print("WARNING: no GPU found, running on CPU (very slow)", file=sys.stderr)
    det = Detector(cfg["weights"], device=device, half=not cfg["fp32"], mode=cfg["detector"])
    # warm-up so cuDNN/CUDA set-up is not counted against the first real video
    import numpy as np

    det.infer([det.preprocess(np.zeros((360, 640, 3), np.uint8))])
    return det


def make_lines(vid, rows, error):
    lines = [json.dumps({"video": vid, "second": s, "objects": o, "people": p}) for s, o, p in rows]
    if error:
        lines.append(json.dumps({"video": vid, "error": error}))
    return lines


def worker_main(wid, task_q, res_q, cfg):
    from collections import defaultdict
    from video_indexer import process_video

    try:
        det = build_detector(cfg, wid)
    except BaseException as e:  # no GPU, weights missing, ...
        res_q.put(("fatal", wid, f"{type(e).__name__}: {e}"))
        return
    res_q.put(("ready", wid))
    totals = defaultdict(float)
    while True:
        task = task_q.get()
        if task is None:
            break
        name, idx, data = task
        vid = f"{name}:{idx}"
        res_q.put(("start", wid, vid))
        stats = defaultdict(float)
        if cfg["detector"] == "fake" and data and data[:7] == b"CRASHME":
            time.sleep(0.5)  # let the "start" message out first
            os._exit(1)  # test hook: simulate a segfault inside the decoder
        try:
            if not data:
                lines = make_lines(vid, [], "empty video")
            else:
                rows, error = process_video(
                    det, data, cfg["k"], cfg["batch"], cfg["decode_threads"],
                    skip_nonref=not cfg["no_skip_nonref"], stats=stats)
                lines = make_lines(vid, rows, error)
                stats["video_seconds"] += len(rows)
        except Exception as e:
            lines = make_lines(vid, [], f"{type(e).__name__}: {e}"[:150])
        for key, val in stats.items():
            totals[key] += val
        res_q.put(("done", wid, vid, lines))
    res_q.put(("exit", wid, dict(totals)))


# --------------------------------------------------------------------------- #
# Reading Parquet (main process, in a thread)
# --------------------------------------------------------------------------- #
def err_line(vid, msg):
    return json.dumps({"video": vid, "error": msg[:150]})


def producer(files, task_q, res_q, state, max_videos, done_videos=frozenset()):
    import pyarrow.parquet as pq

    def fake_done(vid, msg):  # an error that never reaches a worker
        state["produced"] += 1
        res_q.put(("done", -1, vid, [err_line(vid, msg)]))

    n_sent = 0
    for path in files:
        name = path.name
        try:
            pf = pq.ParquetFile(path)
            if "mp4" not in pf.schema_arrow.names:
                raise ValueError("no mp4 column")
        except Exception as e:
            fake_done(name, f"cannot read parquet file: {type(e).__name__}: {e}")
            continue
        row = 0
        for rg in range(pf.num_row_groups):
            nrows = pf.metadata.row_group(rg).num_rows
            try:
                col = pf.read_row_group(rg, columns=["mp4"]).column(0)
            except Exception as e:
                for j in range(nrows):
                    fake_done(f"{name}:{row + j}", f"cannot read row group: {type(e).__name__}")
                row += nrows
                continue
            for j in range(nrows):
                if max_videos and n_sent >= max_videos:
                    state["finished"] = True
                    return
                vid = f"{name}:{row + j}"
                if vid in done_videos:
                    continue
                try:
                    data = col[j].as_py()
                except Exception as e:
                    fake_done(vid, f"cannot read row: {type(e).__name__}")
                    continue
                state["produced"] += 1
                n_sent += 1
                task_q.put((name, row + j, data))  # blocks while all workers are busy
            row += nrows
            del col
    state["finished"] = True


def prepare_resume(path):
    """Videos already in the output. The last video in the file may have been cut short by the
    interruption, so its lines are removed and it is processed again."""
    with open(path, "rb") as f:
        raw = f.read().split(b"\n")[:-1]  # the tail after the last newline is a partial line (or empty)
    keep, vids = [], []
    for line in raw:
        try:
            vids.append(json.loads(line)["video"])
        except Exception:
            break
        keep.append(line)
    last = vids[-1] if vids else None
    while vids and vids[-1] == last:  # drop the whole last block
        vids.pop()
        keep.pop()
    with open(path, "wb") as f:
        for line in keep:
            f.write(line + b"\n")
    return set(vids)


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    t_start = time.perf_counter()
    files = sorted(Path(args.input).rglob("*.parquet")) if Path(args.input).is_dir() else [Path(args.input)]
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print(f"no .parquet files under {args.input}", file=sys.stderr)
        return 2
    cfg = dict(k=args.samples_per_sec, batch=args.batch, decode_threads=args.decode_threads,
               detector=args.detector, fp32=args.fp32, device=args.device,
               weights=args.weights, no_skip_nonref=args.no_skip_nonref)

    ctx = mp.get_context("spawn")  # CUDA and fork don't mix
    task_q = ctx.Queue(maxsize=args.workers * 2)
    res_q = ctx.Queue()
    procs, inflight, exited = {}, {}, set()

    def spawn(wid):
        p = ctx.Process(target=worker_main, args=(wid, task_q, res_q, cfg), daemon=True)
        p.start()
        procs[wid] = p

    for wid in range(args.workers):
        spawn(wid)

    state = {"produced": 0, "finished": False}
    done_videos = set()
    mode = "w"
    if args.resume and os.path.exists(args.output):
        done_videos = prepare_resume(args.output)
        mode = "a"
        print(f"resuming: {len(done_videos)} videos already in {args.output}", file=sys.stderr)
    prod = threading.Thread(target=producer, args=(files, task_q, res_q, state, args.max_videos, done_videos),
                            daemon=True)
    prod.start()

    out = open(args.output, mode)
    completed = n_lines = n_errors = 0
    respawns = 0
    totals = {}
    t_first_done = None
    try:
        while not (state["finished"] and completed >= state["produced"]):
            try:
                msg = res_q.get(timeout=2.0)
            except pyqueue.Empty:
                # did a worker die (segfault, OOM kill, ...)? then its video gets an error line
                for wid, p in list(procs.items()):
                    if wid in exited or p.is_alive():
                        continue
                    exited.add(wid)
                    vid = inflight.pop(wid, None)
                    if vid is not None:
                        out.write(err_line(vid, "worker crashed") + "\n")
                        completed += 1
                        n_errors += 1
                    respawns += 1
                    if respawns > 50:
                        raise RuntimeError("workers keep crashing")
                    print(f"worker {wid} died (video {vid}); starting a new one", file=sys.stderr)
                    spawn(max(procs) + 1)
                continue
            kind = msg[0]
            if kind == "done":
                _, wid, vid, lines = msg
                inflight.pop(wid, None)
                out.write("\n".join(lines) + "\n")
                out.flush()
                completed += 1
                n_lines += len(lines)
                n_errors += sum('"error"' in ln for ln in lines)
                if t_first_done is None:
                    t_first_done = time.perf_counter()
                if completed % 50 == 0:
                    print(f"[{time.perf_counter() - t_start:7.1f}s] {completed} videos done", file=sys.stderr)
            elif kind == "start":
                inflight[msg[1]] = msg[2]
            elif kind == "fatal":
                print(f"worker {msg[1]} cannot start: {msg[2]}", file=sys.stderr)
                raise SystemExit(3)
            elif kind == "exit":
                for key, val in msg[2].items():
                    totals[key] = totals.get(key, 0.0) + val
                exited.add(msg[1])
    finally:
        for _ in range(len(procs)):
            try:
                task_q.put_nowait(None)
            except Exception:
                break
        # collect worker stats while they shut down
        deadline = time.time() + 20
        while len(exited) < len(procs) and time.time() < deadline:
            try:
                msg = res_q.get(timeout=1.0)
            except pyqueue.Empty:
                continue
            if msg[0] == "exit":
                for key, val in msg[2].items():
                    totals[key] = totals.get(key, 0.0) + val
                exited.add(msg[1])
        for p in procs.values():
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        out.close()

    wall = time.perf_counter() - t_start
    vs = totals.get("video_seconds", 0.0)
    print(f"done: {completed} videos, {n_lines} lines ({n_errors} errors) in {wall:.1f}s", file=sys.stderr)
    if args.profile and vs:
        w = max(1, args.workers)
        print(f"  video time processed : {vs / 3600:.2f} h  ({vs / wall:.0f}x real time)", file=sys.stderr)
        print(f"  frames decoded       : {totals.get('frames_decoded', 0):.0f}", file=sys.stderr)
        print(f"  frames through model : {totals.get('frames_inferred', 0):.0f}", file=sys.stderr)
        print("  worker time, summed over workers (s) and as % of wall x workers:", file=sys.stderr)
        for key in ("decode", "preprocess", "infer"):
            v = totals.get(key, 0.0)
            print(f"    {key:<11}{v:9.1f}  {100 * v / (wall * w):5.1f}%", file=sys.stderr)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
