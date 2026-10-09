"""LRE-style baseline through OUR folds: TF-IDF + 6 position/length features ->
L2 logistic regression with balanced class weights, scored by macro-AUC
against gold. A reimplementation from the paper's description (LRE's exact
TF-IDF settings and regularisation strength are not known to us).

Usage: python scripts/experiments/lr_baseline.py {locomo|lme} {label_supervised|label_overlap}
"""

import math
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from aamcf_data.gate.dataset import LME, LOCOMO, load_units, split_folds

ds, label = sys.argv[1], sys.argv[2]
cfg, cv = (LOCOMO, None) if ds == "locomo" else (LME, 5)
df = load_units(cfg, label)
gold = load_units(cfg, "label_supervised")["label"].to_numpy()

n = df.groupby("group")["text"].transform("size").to_numpy()
idx = df.groupby("group").cumcount().to_numpy()
t = df["text"]
traj = np.column_stack([
    idx / np.maximum(1, n),
    1.0 / (1 + (n - 1 - idx)),
    np.log1p(t.str.split().str.len().to_numpy()),
    t.str.count(r"\d").to_numpy(),
    t.str.contains(r"\?", regex=True).astype(float).to_numpy(),
    t.apply(lambda s: sum(1 for w in s.split() if w[:1].isupper())).to_numpy(),
])

aucs = []
for k, (tr, te, _) in enumerate(split_folds(df, cv)):
    vec = TfidfVectorizer(max_features=20000, min_df=2, sublinear_tf=True)
    sc = StandardScaler().fit(traj[tr])
    Xtr = hstack([vec.fit_transform(t.iloc[tr]), csr_matrix(sc.transform(traj[tr]))]).tocsr()
    Xte = hstack([vec.transform(t.iloc[te]), csr_matrix(sc.transform(traj[te]))]).tocsr()
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, df["label"].to_numpy()[tr])
    out = pd.DataFrame({"group": df["group"].iloc[te].to_numpy(), "label": gold[te], "score": clf.predict_proba(Xte)[:, 1]})
    a = [roc_auc_score(g["label"], g["score"]) for _, g in out.groupby("group") if g["label"].nunique() == 2]
    aucs.append(float(np.mean(a)))
    print(f"fold {k}: macro-AUC {aucs[-1]:.3f} over {len(a)} groups", flush=True)
print(f"RESULT LR baseline {ds} {label}: mean macro-AUC {np.mean(aucs):.3f} over {len(aucs)} folds")
