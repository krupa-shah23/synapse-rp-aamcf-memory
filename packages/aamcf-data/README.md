# aamcf-data — Stage 0a candidate-memory gate

```
src/aamcf_data/
  labels/overlap.py        LRE's answer-recall proxy label
  locomo/                  parser.py (turns, QA, evidence) + labels.py
  longmemeval/             parser.py (sessions) + labels.py
  gate/dataset.py          units, windows, leakage-free splits
  gate/train.py            DistilBERT gate: CV fold, or --final to save a model
scripts/
  labels/                  download data, build label parquets
  experiments/             fold drivers, LR baseline, Kaggle packaging
  inspect/                 peek at training rows
data/                      (git-ignored)
  locomo/, longmemeval/    raw json + label parquets
  gate/                    <dataset>_<label>_fold<k>.parquet = 30 CV score files
  gate/final_<dataset>_<label>/   saved gate: model.safetensors, tokenizer, gate_meta.json (threshold)
```
Run all commands from `packages/aamcf-data` with `uv run python ...`.
