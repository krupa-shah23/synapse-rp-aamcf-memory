"""Stage 0a gate: fine-tune DistilBERT (2 classes: 0 = not memory, 1 = candidate)
on one fold and report LRE's metric, macro-AUC per group.

LoCoMo unit = turn; LongMemEval_S unit = session, scored as the max over its
windows. Windows inherit the session label, and negatives are subsampled per
epoch (neg_ratio x positives) to handle the ~4% positive rate cheaply.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from aamcf_data.gate.dataset import LME, LME_SMALL, LOCOMO, load_units, make_windows, split_folds

MODEL = os.environ.get("GATE_MODEL", "distilbert-base-uncased")  # local path on Kaggle (no Internet)


def pad(batch_ids: list[list[int]], device) -> dict:
    n = max(map(len, batch_ids))
    ids = torch.zeros(len(batch_ids), n, dtype=torch.long)
    mask = torch.zeros(len(batch_ids), n, dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}


@torch.no_grad()
def score_windows(model, ids, device, bs=64) -> np.ndarray:
    model.eval()
    order = np.argsort([len(x) for x in ids])  # similar lengths per batch
    out = np.zeros(len(ids), dtype=np.float32)
    for s in range(0, len(ids), bs):
        b = order[s : s + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**pad([ids[i] for i in b], device)).logits
        out[b] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return out


def macro_auc(df: pd.DataFrame) -> tuple[float, int]:
    aucs = [roc_auc_score(g["label"], g["score"]) for _, g in df.groupby("group") if g["label"].nunique() == 2]
    return float(np.mean(aucs)), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["locomo", "lme", "lme_small"], required=True)
    ap.add_argument("--label", choices=["label_supervised", "label_overlap"], default="label_supervised")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda"
    cfg, cv = {"locomo": (LOCOMO, None), "lme": (LME, 5), "lme_small": (LME_SMALL, 5)}[a.dataset]
    df = load_units(cfg, a.label)
    gold = load_units(cfg, "label_supervised")["label"].to_numpy()  # evaluation is always against gold, as in LRE
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = make_windows(tok, df["text"].tolist())
    ids, unit = enc["input_ids"], np.array(enc["unit_idx"])
    wlabel = df["label"].to_numpy()[unit]

    tr, te, removed = list(split_folds(df, cv))[a.fold]
    tr_w, te_w = np.flatnonzero(np.isin(unit, tr)), np.flatnonzero(np.isin(unit, te))
    pos_w, neg_w = tr_w[wlabel[tr_w] == 1], tr_w[wlabel[tr_w] == 0]
    print(f"fold {a.fold}: train units {len(tr)} (dedup removed {removed}), test units {len(te)} | train windows pos {len(pos_w)} neg {len(neg_w)} | test windows {len(te_w)}")

    model = DistilBertForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    for ep in range(a.epochs):
        k = min(len(neg_w), int(a.neg_ratio * len(pos_w))) if len(pos_w) else len(neg_w)
        epoch = rng.permutation(np.concatenate([pos_w, rng.choice(neg_w, k, replace=False)]))
        model.train()
        total, t0 = 0.0, time.time()
        for s in range(0, len(epoch), a.bs):
            b = epoch[s : s + a.bs]
            y = torch.tensor(wlabel[b], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(**pad([ids[i] for i in b], device)).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(b)
            if (s // a.bs) % 100 == 0:
                print(f"  step {s // a.bs}/{len(epoch) // a.bs} loss {loss.item():.3f} {(s + len(b)) / (time.time() - t0):.1f} win/s", flush=True)
        print(f"epoch {ep + 1}: {len(epoch)} windows, mean loss {total / len(epoch):.4f}")

    ws = score_windows(model, [ids[i] for i in te_w], device)
    res = pd.DataFrame({"unit": unit[te_w], "score": ws}).groupby("unit")["score"].max()  # session score = max over windows
    out = df.iloc[res.index].copy()
    out["score"] = res.to_numpy()
    out["label"] = gold[res.index]
    auc, n = macro_auc(out)
    print(f"RESULT {a.dataset} {a.label} fold {a.fold}: macro-AUC {auc:.3f} over {n} groups | pooled AUC {roc_auc_score(out['label'], out['score']):.3f}")
    p = Path(f"data/gate/{a.dataset}_{a.label}_fold{a.fold}.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    out[["group", "key", "label", "score"]].to_parquet(p, index=False)


if __name__ == "__main__":
    main()
