"""Combine label_supervised, label_overlap, and label_distant into one
training-ready label table for LongMemEval_S sessions — the Y-labels for
the Stage 0a DistilBERT gate classifier. No model touches this file.
"""

from __future__ import annotations

import math

import pandas as pd

from aamcf_data.labels.overlap import label_overlap
from aamcf_data.longmemeval.parser import (
    build_evidence_index,
    flatten_questions,
    flatten_sessions,
    load_raw,
)

LAMBDA_DAYS = 30  # decay constant in days, tunable like LoCoMo's LAMBDA_QA


def compute_label_supervised(sessions_df: pd.DataFrame, evidence_index: dict) -> pd.Series:
    return sessions_df.apply(
        lambda row: int(f"{row['question_id']}::{row['session_id']}" in evidence_index),
        axis=1,
    )


def compute_label_overlap(sessions_df: pd.DataFrame, questions_df: pd.DataFrame, tau: float = 0.4) -> pd.Series:
    """LRE's annotation-free proxy: the whole session (role-prefixed text,
    as LRE builds it) against the question's gold answer, answer-side recall
    >= tau. Deliberately independent of answer_session_ids and has_answer —
    has_answer is set only inside gold sessions, so using it here would make
    this label a subset of label_supervised."""
    answers_by_question = questions_df.set_index("question_id")["answer_text"].to_dict()
    return sessions_df.apply(
        lambda row: label_overlap(
            row["session_text_roles"], [answers_by_question.get(row["question_id"], "")], tau=tau
        ),
        axis=1,
    )


def compute_label_distant(
    sessions_df: pd.DataFrame, questions_df: pd.DataFrame, evidence_index: dict, lam: float = LAMBDA_DAYS
) -> pd.Series:
    """Mirrors LoCoMo's compute_label_distant: nonzero only for rows already
    in evidence_index, never spread to a different session by proximity —
    that fabricates a relevance signal for rows with no evidence of their
    own. dt uses real calendar dates (question_date vs. session_date)
    rather than LoCoMo's session-index positions, since LongMemEval
    provides actual timestamps for both."""
    question_dates = pd.to_datetime(questions_df.set_index("question_id")["question_date"])

    def score(row):
        if f"{row['question_id']}::{row['session_id']}" not in evidence_index:
            return 0.0
        dt = abs((question_dates[row["question_id"]] - pd.Timestamp(row["session_date"])).days)
        return 1 - math.exp(-math.exp(-dt / lam))

    return sessions_df.apply(score, axis=1)


def build_label_table(raw_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    questions = load_raw(raw_path)
    sessions_df = flatten_sessions(questions)
    questions_df = flatten_questions(questions)
    evidence_index = build_evidence_index(questions_df)

    sessions_df["label_supervised"] = compute_label_supervised(sessions_df, evidence_index)
    sessions_df["label_overlap"] = compute_label_overlap(sessions_df, questions_df)
    sessions_df["label_distant"] = compute_label_distant(sessions_df, questions_df, evidence_index)

    # Union of sources, not a weighted blend — per spec: let the gate see
    # the union, don't hand-weight the sources.
    sessions_df["label"] = (
        (sessions_df["label_supervised"] == 1)
        | (sessions_df["label_overlap"] == 1)
        | (sessions_df["label_distant"] >= 0.5)
    ).astype(int)

    return sessions_df, questions_df, evidence_index
