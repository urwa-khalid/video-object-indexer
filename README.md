# Video indexing at speed

For every second of every video in a folder of FineVideo Parquet files, `run.py` writes which of
`person, car, dog, laptop, cell phone` appear and how many people are visible (Ultralytics YOLO11l,
input size 640, confidence >= 0.4, NMS IoU 0.7).

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Requirements

Ubuntu 22.04, Python 3.11, an NVIDIA GPU with a driver that supports CUDA 12.

## Install

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0     # OpenCV runtime libraries
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation (needs internet, not timed)

```bash
python download_weights.py
```

Downloads the official `yolo11l.pt` next to `run.py` and checks that PyTorch can use every GPU.
`run.py` itself needs no internet.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
gzip -k results.jsonl        # results.jsonl.gz
```

The folder is searched recursively for `*.parquet`. Progress is printed to stderr.
If a run is interrupted, start it again with `--resume` (same `--output`) and finished videos are skipped.

## Output

One JSON line per second of each video. `video` is the Parquet file name, a colon and the row number:

```
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
{"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
```

## Options

| option | default | meaning |
| --- | --- | --- |
| `--samples-per-sec K` | 4 | frames per second sent to the model. `0` = every frame (reference mode) |
| `--workers N` | CPU cores / 2 (2 to 16) | worker processes; each decodes videos and owns a copy of the model |
| `--decode-threads N` | 2 | ffmpeg threads per worker |
| `--batch N` | 16 | frames per model call |
| `--device` | auto | `auto` spreads workers over all GPUs; or `cpu`, `cuda:0`, ... |
| `--fp32` | off | float32 instead of float16 |
| `--detector` | fast | `fast` = own forward pass + Ultralytics NMS, `official` = `YOLO.predict()` |
| `--profile` | off | print where the time goes |
| `--resume` | off | skip videos already in `--output` |

`python run.py --help` lists everything.

## How it works

1. The main process reads each Parquet file row by row (one mp4 per row) and queues the bytes.
2. Each worker process decodes a video with PyAV (ffmpeg) on the CPU. Seconds are counted from the
   first decoded frame; the video's rotation tag is applied. Only `K` frames per second are kept
   (each second is cut into K equal slots and the first frame of each slot is used), and frames
   that no other frame depends on are not decoded at all.
3. The kept frames are letterboxed like Ultralytics does and run through YOLO11l in batches (FP16),
   with Ultralytics NMS. Only the five classes are counted.
4. Per second: an object is present if it is found in at least half of the kept frames; the people
   count is the median of the per-frame person counts, rounded down when halfway. The output covers
   every second up to the one holding the last frame.
5. A video that cannot be decoded gets one `error` line. If it breaks part way, the lines already
   produced are kept and one `error` line is added. A worker that crashes costs one error line, not the run.

## Checking correctness

```bash
# reference: every frame, official Ultralytics predict(), FP32
python run.py --input <folder> --output ref.jsonl --max-videos 12 --samples-per-sec 0 --detector official --fp32 --no-skip-nonref
# fast run on the same videos
python run.py --input <folder> --output fast.jsonl --max-videos 12
python tools/evaluate_results.py ref.jsonl fast.jsonl     # object F1, people exact / within 1
python tools/verify_detector.py --video some.mp4          # fast detector == official YOLO.predict ?
python -m pytest -q tests                                 # logic tests, no GPU needed
```

## My run

Full run on 2 x Tesla T4 (Kaggle): 1650 videos, 128.27 h of video, 461,757 output lines, 0 errors, 12,590 s
(see `HARDWARE.md`). The notebooks used are in `notebooks/`.

Sampling rate vs the every-frame reference (12 videos from each of the first two Parquet files, about 1 h of video each).
Columns: object F1 (micro) / people exact / people within 1. The brief needs 0.95 / 0.85 / 0.95.

| frames per second | file 00000 | file 00001 |
| --- | --- | --- |
| 2 | 0.9630 / 0.9141 / 0.9913 | 0.9642 / 0.9135 / 0.9929 |
| 3 | 0.9815 / 0.9491 / 0.9940 | 0.9811 / 0.9466 / 0.9943 |
| **4 (used)** | 0.9835 / 0.9523 / 0.9949 | 0.9806 / 0.9488 / 0.9992 |
| 6 | 0.9876 / 0.9591 / 0.9965 | 0.9871 / 0.9631 / 0.9989 |

Note: `tools/evaluate_results.py` also checks the per-class (macro) F1 and prints FAIL unless that is >= 0.95 as well.
For 4 frames per second on these 24 videos it was below 0.95, even though the three thresholds above pass.

## Files

- `run.py`, `video_indexer/` : the script (decoding, detector, per-second rule)
- `download_weights.py` : preparation command
- `tools/` : result scoring and detector check; `tests/` : unit tests
- `results.jsonl.gz` : results for the first 50 Parquet files
- `HARDWARE.md`, `WRITEUP.md`, `ai_chats/` : hardware description, write-up, AI chat history
- `notebooks/` : Kaggle notebooks for the accuracy check and the full run
