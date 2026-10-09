#!/bin/bash
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
for f in 1 2 3 4 5 6 7 8 9; do uv run python -m aamcf_data.gate.train --dataset locomo --fold $f > data/gate/locomo_gold_fold$f.log 2>&1; done
for f in 1 2 3 4; do uv run python -m aamcf_data.gate.train --dataset lme --fold $f > data/gate/lme_gold_fold$f.log 2>&1; done
echo ALL_DONE > data/gate/gold_folds.done
