# Hardware used for the full 50-file run

Fill in from the machine where you ran all 50 files (commands in the comments).

- Where I ran it: <own computer / university / Colab / Kaggle / rented GPU (provider)>
- GPU model and memory: <nvidia-smi --query-gpu=name,memory.total --format=csv>
- CPU model and cores: <lscpu | grep -E "Model name|^CPU\(s\)">
- RAM: <free -h>
- Disk type: <SSD or hard disk; lsblk -d -o NAME,ROTA  (ROTA 0 = SSD)>
- Operating system: <lsb_release -a>
- Python / torch / ultralytics versions: <pip freeze | grep -iE "^(torch|ultralytics|av)=">
- Command used: <python run.py --input finevideo/data --output results.jsonl ...>
- Time for the full run, start to finish: <from the "done: ... in N s" line of run.py>
