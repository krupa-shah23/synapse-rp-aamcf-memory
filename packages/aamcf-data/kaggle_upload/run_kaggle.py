"""Run LongMemEval gate folds on Kaggle. Usage in a notebook cell:
    !python /kaggle/input/<dataset-name>/run_kaggle.py proxy 1 2 3 4
First arg: gold or proxy. Rest: fold numbers. Folds are spread over the GPUs."""
import os, shutil, subprocess, sys
from pathlib import Path

import torch

src = Path(__file__).parent
work = Path("/kaggle/working")
for d in ["src", "data", "folds_lme.json", "distilbert-base-uncased"]:
    dst = work / d
    if not dst.exists():
        (shutil.copytree if (src / d).is_dir() else shutil.copy)(src / d, dst)
label = {"gold": "label_supervised", "proxy": "label_overlap"}[sys.argv[1]]
folds = [int(x) for x in sys.argv[2:]]
n = max(1, torch.cuda.device_count())
print("GPUs:", n, "| folds:", folds, "| label:", label, flush=True)
env = dict(os.environ, PYTHONPATH=str(work / "src"), GATE_MODEL=str(work / "distilbert-base-uncased"),
           GATE_FOLDS=str(work / "folds_lme.json"), PYTHONIOENCODING="utf-8")
workers = []
for g in range(n):
    mine = folds[g::n]
    cmd = " && ".join(
        f"python -m aamcf_data.gate.train --dataset lme --label {label} --fold {f} > data/gate/lme_{sys.argv[1]}_fold{f}.log 2>&1" for f in mine)
    if cmd:
        workers.append(subprocess.Popen(cmd, shell=True, cwd=work, env=dict(env, CUDA_VISIBLE_DEVICES=str(g))))
for w in workers:
    w.wait()
print("DONE. Download /kaggle/working/data/gate/*", flush=True)
