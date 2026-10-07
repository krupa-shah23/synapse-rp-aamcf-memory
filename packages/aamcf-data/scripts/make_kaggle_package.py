"""Build kaggle_upload/ (and a zip): gate code, slim LongMemEval labels, fixed
fold list, local DistilBERT weights, and the notebook runner."""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from aamcf_data.gate.dataset import LME, load_units, split_folds

OUT = Path("kaggle_upload")
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "src/aamcf_data/gate").mkdir(parents=True)
(OUT / "data/longmemeval").mkdir(parents=True)
(OUT / "data/gate").mkdir(parents=True)

for f in ["__init__.py"]:
    shutil.copy(f"src/aamcf_data/{f}", OUT / f"src/aamcf_data/{f}")
for f in ["__init__.py", "dataset.py", "train.py"]:
    shutil.copy(f"src/aamcf_data/gate/{f}", OUT / f"src/aamcf_data/gate/{f}")

cols = ["question_id", "session_id", "session_text_roles", "label_supervised", "label_overlap"]
df = pd.read_parquet(LME["path"], columns=cols)
df.to_parquet(OUT / "data/longmemeval/longmemeval_labels.parquet", index=False)

units = load_units(LME, "label_supervised")
folds = {}
for k, (tr, te, _) in enumerate(split_folds(units, 5)):
    folds[str(k)] = sorted(set(units["group"].iloc[te]))
json.dump(folds, open(OUT / "folds_lme.json", "w"))

snapshot_download("distilbert-base-uncased", local_dir=OUT / "distilbert-base-uncased",
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"])

(OUT / "run_kaggle.py").write_text('''"""Run LongMemEval gate folds on Kaggle. Usage in a notebook cell:
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
''', encoding="utf-8")

zp = Path("kaggle_upload.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT))
print("folds:", {k: len(v) for k, v in folds.items()}, "| zip MB:", round(zp.stat().st_size / 1e6))
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(round(p.stat().st_size / 1e6, 1), "MB", p.relative_to(OUT))
