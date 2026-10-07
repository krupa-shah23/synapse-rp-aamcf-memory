"""Combine label_supervised, label_overlap, and label_distant into one
training-ready label table for LoCoMo turns — the Y-labels for the
Stage 0a DistilBERT gate classifier. No model touches this file.
"""

from __future__ import annotations

import math

import pandas as pd

from aamcf_data.labels.overlap import label_overlap
from aamcf_data.locomo.parser import build_evidence_index, flatten_qa, flatten_turns, load_raw

LAMBDA_QA = 5  # decay constant in session-index units (approximation — LoCoMo
                # has no per-question timestamp, only session order)


def compute_label_supervised(turns_df: pd.DataFrame, evidence_index: dict) -> pd.Series:
    return turns_df.apply(
        lambda row: int(f"{row['conversation_id']}::{row['turn_id']}" in evidence_index),
        axis=1,
    )


def compute_label_distant(turns_df: pd.DataFrame, evidence_index: dict, qid_to_session: dict, lam: float = LAMBDA_QA) -> pd.Series:
    def score(row):
        key = f"{row['conversation_id']}::{row['turn_id']}"
        qids = evidence_index.get(key, [])
        if not qids:
            return 0.0
        total = sum(
            math.exp(-abs(qid_to_session.get(qid, row["session_num"]) - row["session_num"]) / lam)
            for qid in qids
        )
        return 1 - math.exp(-total)
    return turns_df.apply(score, axis=1)


def compute_label_overlap(turns_df: pd.DataFrame, qa_df: pd.DataFrame, tau: float = 0.4) -> pd.Series:
    # LRE drops adversarial (category 5) questions; they carry no gold answer.
    answered = qa_df[qa_df["category"] != 5]
    answers_by_conv = answered.groupby("conversation_id")["answer_text"].apply(list).to_dict()
    return turns_df.apply(
        lambda row: label_overlap(row["overlap_text"], answers_by_conv.get(row["conversation_id"], []), tau=tau),
        axis=1,
    )


def build_label_table(raw_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    conversations = load_raw(raw_path)
    turns_df = flatten_turns(conversations)
    qa_df = flatten_qa(conversations)
    evidence_index = build_evidence_index(qa_df)

    qid_to_session: dict[str, int] = {}
    for _, row in turns_df.iterrows():
        key = f"{row['conversation_id']}::{row['turn_id']}"
        for qid in evidence_index.get(key, []):
            qid_to_session.setdefault(qid, row["session_num"])

    turns_df["label_supervised"] = compute_label_supervised(turns_df, evidence_index)
    turns_df["label_overlap"] = compute_label_overlap(turns_df, qa_df)
    turns_df["label_distant"] = compute_label_distant(turns_df, evidence_index, qid_to_session)

    # Union of sources, not a weighted blend — per spec: let the gate see
    # the union, don't hand-weight the sources.
    turns_df["label"] = (
        (turns_df["label_supervised"] == 1)
        | (turns_df["label_overlap"] == 1)
        | (turns_df["label_distant"] >= 0.5)
    ).astype(int)

    return turns_df, qa_df, evidence_index