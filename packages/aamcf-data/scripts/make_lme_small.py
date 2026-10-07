"""Smoke-test subset of LongMemEval_S: N questions sampled proportionally by
question_type, each kept with its full haystack so labels stay valid."""

import sys

import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
q = pd.read_parquet("data/longmemeval/longmemeval_questions.parquet", columns=["question_id", "question_type"])
pick = (
    q.groupby("question_type", group_keys=False)
    .apply(lambda g: g.sample(max(1, round(N * len(g) / len(q))), random_state=0))["question_id"]
)
df = pd.read_parquet("data/longmemeval/longmemeval_labels.parquet")
small = df[df["question_id"].isin(set(pick))].reset_index(drop=True)
small.to_parquet("data/longmemeval/longmemeval_labels_small.parquet", index=False)
print(f"questions {small['question_id'].nunique()} | sessions {len(small)} | gold {small['label_supervised'].mean():.2%} | proxy {small['label_overlap'].mean():.2%}")
print(q[q["question_id"].isin(set(pick))]["question_type"].value_counts().to_string())
