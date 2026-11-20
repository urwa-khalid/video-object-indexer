# Video indexing at speed

For every second of every video in a folder of FineVideo Parquet files: which of
`person, car, dog, laptop, cell phone` appear (YOLO11l, 640, conf >= 0.4) and how many people are visible.

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0     # OpenCV runtime libraries
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_weights.py   # preparation command: downloads the official yolo11l.pt (needs internet, not timed)
```

## Run

```bash
python run.py --input finevideo/data --output results.jsonl
```

Useful options (all have defaults): `--samples-per-sec`, `--workers`, `--batch`, `--decode-threads`,
`--fp32`, `--profile` (prints where the time goes), `--resume` (continue an interrupted run into the same
`--output`). With several GPUs the workers are spread over all of them automatically.
`python run.py --help` lists everything.

## What it does

1. The main process reads every `.parquet` file row by row (one mp4 per row) and queues the bytes.
2. N worker processes each take a video, decode it with PyAV (ffmpeg) on the CPU and keep a few frames
   per second (`--samples-per-sec`), letterbox them like Ultralytics does, and run YOLO11l on the GPU in batches.
3. Per second: an object counts if it is found in at least half of the kept frames; people = median
   of the per-frame person counts, rounded down. Seconds are counted from the first decoded frame and the
   output covers every second up to the one holding the last frame.
4. A damaged video gives an `error` line (lines already produced for it are kept); a crashed worker costs
   one error line, not the run.

## Checking results (tools)

```bash
# slow reference: every frame, official Ultralytics predict(), fp32 (the brief's definition of "correct")
python run.py --input <folder> --output ref.jsonl --samples-per-sec 0 --detector official --fp32 --no-skip-nonref
python run.py --input <folder> --output fast.jsonl
python tools/evaluate_results.py ref.jsonl fast.jsonl   # object F1, people exact / within 1, PASS/FAIL
python tools/verify_detector.py --video some.mp4      # fast detector path == official YOLO.predict ?
python -m pytest -q tests                           # logic tests, no GPU needed
```
