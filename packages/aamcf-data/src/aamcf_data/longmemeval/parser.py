"""Flatten longmemeval_s_cleaned.json into canonical session + question tables.

Source: LongMemEval_S, data/longmemeval/longmemeval_s_cleaned.json
question ->
    question_id, question_type, question, answer, question_date
    haystack_session_ids: list[str], parallel to haystack_sessions/haystack_dates
    haystack_dates: list[str], parallel to haystack_session_ids
    haystack_sessions: list[list[turn]], turn = {role, content, has_answer?}
    answer_session_ids: list[str], gold evidence session ids for this question

LRE's Appendix B.2 builds LongMemEval_S labels at session granularity (not
turn-level), which this module follows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from dateutil import parser as dateparser


def load_raw(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_sessions(questions: list[dict]) -> pd.DataFrame:
    """One row per (question_id, session_id) pair.

    session_id is NOT globally unique across questions — the same session
    is reused as haystack filler across many different questions (LRE's own
    preprocessing needed a dedup guard for this, dropping 51,000+ duplicate
    session instances). So every row here is scoped to its own question;
    see build_evidence_index for the composite key this implies.
    """
    rows = []
    for q in questions:
        question_id = q["question_id"]
        session_ids = q["haystack_session_ids"]
        dates = q["haystack_dates"]
        sessions = q["haystack_sessions"]
        for idx, (session_id, date_str, turns) in enumerate(zip(session_ids, dates, sessions)):
            session_ts = dateparser.parse(date_str, fuzzy=True) if date_str else None
            session_text = "\n".join(turn.get("content", "") or "" for turn in turns)
            answer_turn_text = "\n".join(
                turn.get("content", "") or "" for turn in turns if turn.get("has_answer")
            )
            rows.append({
                "question_id": question_id,
                "session_id": session_id,
                "session_index": idx,
                "session_date": session_ts.isoformat() if session_ts else None,
                "session_text": session_text,
                "answer_turn_text": answer_turn_text,
                "num_turns": len(turns),
            })
    return pd.DataFrame(rows)


def flatten_questions(questions: list[dict]) -> pd.DataFrame:
    """One row per question."""
    rows = []
    for q in questions:
        date_str = q.get("question_date")
        question_date = dateparser.parse(date_str, fuzzy=True) if date_str else None
        rows.append({
            "question_id": q["question_id"],
            "question_type": q.get("question_type"),
            "question_text": q.get("question", ""),
            "answer_text": str(q.get("answer", "")),
            "question_date": question_date.isoformat() if question_date else None,
            "answer_session_ids": q.get("answer_session_ids", []),
        })
    return pd.DataFrame(rows)


def build_evidence_index(questions_df: pd.DataFrame) -> dict[str, int]:
    """(question_id, session_id) composite key -> 1 if session_id is gold
    evidence for that question. Composite key because session_id repeats
    across questions' haystacks — a session that's gold evidence for
    question A can appear as plain filler in question B's haystack, so
    session_id alone would mislabel it (the LoCoMo dia_id collision bug,
    recurring here). Unlike LoCoMo's evidence index, this is a direct
    per-question membership check, not an aggregated citation list —
    LongMemEval's gold is per-question binary, not shared across questions.
    """
    index: dict[str, int] = {}
    for _, row in questions_df.iterrows():
        question_id = row["question_id"]
        for session_id in row["answer_session_ids"]:
            index[f"{question_id}::{session_id}"] = 1
    return index
