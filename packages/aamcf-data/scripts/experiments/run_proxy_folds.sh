#!/bin/bash
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
for f in 0 1 2 3 4 5 6 7 8 9; do [ -f data/gate/locomo_label_overlap_fold$f.parquet ] || uv run python -m aamcf_data.gate.train --dataset locomo --label label_overlap --fold $f > data/gate/locomo_proxy_fold$f.log 2>&1; done
for f in 0 1 2 3 4; do [ -f data/gate/lme_label_overlap_fold$f.parquet ] || uv run python -m aamcf_data.gate.train --dataset lme --label label_overlap --fold $f > data/gate/lme_proxy_fold$f.log 2>&1; done
echo ALL_DONE > data/gate/proxy_folds.done
