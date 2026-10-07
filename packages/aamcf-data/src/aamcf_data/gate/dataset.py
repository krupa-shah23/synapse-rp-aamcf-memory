"""Stage 0a gate inputs: units, labels, windows and leakage-free splits.

Units follow LRE: LoCoMo = one turn, LongMemEval_S = one whole session.
Sessions are longer than DistilBERT's 512-token limit, so a session is cut
into overlapping windows; the model scores each window and the session score
is the max over its windows. Splits follow LRE: leave-one-conversation-out
on LoCoMo, grouped 5-fold by question on LongMemEval_S, with an exact-text
dedup guard that drops training units whose text appears in the test fold.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

MAX_LEN = 512
STRIDE = 128  # token overlap between consecutive windows

LOCOMO = {"path": "data/locomo/locomo_labels.parquet", "text": "text", "group": "conversation_id", "key": "turn_id"}
LME = {"path": "data/longmemeval/longmemeval_labels.parquet", "text": "session_text_roles", "group": "question_id", "key": "session_id"}
LME_SMALL = {**LME, "path": "data/longmemeval/longmemeval_labels_small.parquet"}


def load_units(cfg: dict, label_col: str) -> pd.DataFrame:
    """One row per unit: text, group, 0/1 label. label_col is label_supervised
    (gold) or label_overlap (LRE proxy), not the union `label`."""
    df = pd.read_parquet(cfg["path"], columns=[cfg["group"], cfg["key"], cfg["text"], label_col])
    df = df.rename(columns={cfg["text"]: "text", cfg["group"]: "group", cfg["key"]: "key", label_col: "label"})
    df["text"] = df["text"].fillna("")
    return df.reset_index(drop=True)


def make_windows(tokenizer, texts: list[str], max_len: int = MAX_LEN, stride: int = STRIDE) -> dict:
    """Cut each text into overlapping windows of at most max_len tokens
    (with [CLS]/[SEP]). Done by hand: the tokenizer's own overflow handling
    returns only the first window plus one leftover chunk, dropping the rest
    of a long session. Returns `input_ids` (one list per window) and
    `unit_idx`, the index of the unit each window came from."""
    body, step = max_len - 2, max_len - 2 - stride
    all_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    input_ids, unit_idx = [], []
    for i, ids in enumerate(all_ids):
        starts = range(0, max(len(ids) - stride, 1), step)
        for s in starts:
            input_ids.append([tokenizer.cls_token_id] + ids[s : s + body] + [tokenizer.sep_token_id])
            unit_idx.append(i)
    return {"input_ids": input_ids, "unit_idx": unit_idx}


def split_folds(df: pd.DataFrame, cv_splits: int | None):
    """Yield (train_idx, test_idx) per fold with the dedup guard applied.
    cv_splits=None -> leave-one-group-out (LoCoMo); else grouped K-fold."""
    groups = df["group"].to_numpy()
    splitter = GroupKFold(n_splits=min(cv_splits, len(set(groups)))) if cv_splits else LeaveOneGroupOut()
    hashes = np.array([int(hashlib.md5(t.encode("utf-8")).hexdigest()[:15], 16) for t in df["text"]])
    folds = splitter.split(df, df["label"], groups)
    if os.environ.get("GATE_FOLDS"):  # fixed test groups per fold, so other machines reproduce the same split
        fixed = json.load(open(os.environ["GATE_FOLDS"], encoding="utf-8"))
        folds = [(np.flatnonzero(~np.isin(groups, fixed[str(k)])), np.flatnonzero(np.isin(groups, fixed[str(k)]))) for k in range(len(fixed))]
    for tr, te in folds:
        leaked = np.isin(hashes[tr], hashes[te])
        yield tr[~leaked], te, int(leaked.sum())
