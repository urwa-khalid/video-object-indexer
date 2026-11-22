# Hardware used for the full 50-file run

- Where I ran it: Kaggle notebook (Save Version -> Save & Run All), accelerator "GPU T4 x2".
- GPU model and memory: 2 x NVIDIA Tesla T4, 15360 MiB each (driver 580.178.04). Both GPUs were used; the worker processes are spread over them.
- CPU model and cores: Intel(R) Xeon(R) CPU @ 2.00GHz, 4 vCPUs (`nproc` = 4).
- RAM: 31 GB.
- Disk type: SSD (`lsblk` ROTA = 0). The 50 Parquet files (24.8 GB) were on `/tmp`, the output on `/kaggle/working` (20 GB).
- Operating system: Ubuntu 24.04.4 LTS (Kaggle image). The script ran inside a Python 3.11.18 virtual environment created with `uv` (the Kaggle system Python is 3.13.16).
- Python / torch / ultralytics versions: Python 3.11.18, torch 2.8.0+cu128, ultralytics 8.4.216, av 18.1.0 (full list in `requirements.txt`).
- Command used: `python run.py --input /tmp/finevideo/data --output /kaggle/working/results.jsonl --profile` (default settings: 4 sampled frames per second, 2 worker processes, batch 16, FP16).
- Time for the full run, start to finish: 12,590 s = 3 h 30 min for `run.py` (log line: `done: 1650 videos, 461757 lines (0 errors) in 12589.6s`). This is 128.27 hours of video, about 37x real time. Downloading the files (about 2 min) and installing the packages are not included.
