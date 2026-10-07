"""Flatten locomo10.json into canonical turn + QA tables.

Source: snap-research/locomo, data/locomo10.json.
conversation["session_N"]           -> list of turns {speaker, dia_id, text, img_url?, blip_caption?}
conversation["session_N_date_time"] -> free-text date, e.g. "1:56 pm on 8 May, 2023"
qa                                   -> [{question, answer, evidence: [dia_id, ...], category}]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dateutil import parser as dateparser


def load_raw(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_turns(conversations: list[dict]) -> pd.DataFrame:
    """One row per dialogue turn (dia_id), across all conversations/sessions."""
    rows = []
    for conv in conversations:
        conv_id = conv["sample_id"]
        c = conv["conversation"]
        session_keys = sorted(
            (k for k in c if k.startswith("session_") and not k.endswith("_date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        for skey in session_keys:
            session_num = int(skey.split("_")[1])
            date_str = c.get(f"{skey}_date_time")
            session_ts = dateparser.parse(date_str) if date_str else None
            for idx, turn in enumerate(c[skey]):
                is_image = bool(turn.get("img_url"))
                text = turn.get("blip_caption") if is_image else turn.get("text")
                rows.append({
                    "dataset": "locomo",
                    "conversation_id": conv_id,
                    "session_id": f"{conv_id}_s{session_num}",
                    "session_num": session_num,
                    "session_timestamp": session_ts.isoformat() if session_ts else None,
                    "turn_id": turn["dia_id"],
                    "speaker": turn.get("speaker"),
                    "text": text,
                    "overlap_text": turn.get("text") or (f"[image: {turn['blip_caption']}]" if turn.get("blip_caption") else ""),
                    "modality": "image" if is_image else "text",
                    "turn_index_in_session": idx,
                })
    return pd.DataFrame(rows)


def flatten_qa(conversations: list[dict]) -> pd.DataFrame:
    """One row per QA pair, with its evidence turn_ids."""
    rows = []
    for conv in conversations:
        conv_id = conv["sample_id"]
        for i, qa in enumerate(conv.get("qa", [])):
            rows.append({
                "dataset": "locomo",
                "conversation_id": conv_id,
                "question_id": f"{conv_id}_q{i}",
                "question_text": qa.get("question", ""),
                "answer_text": str(qa.get("answer", "")),
                "category": qa.get("category"),
                "evidence_turn_ids": qa.get("evidence", []),
            })
    return pd.DataFrame(rows)


def build_evidence_index(qa_df: pd.DataFrame) -> dict[str, list[str]]:
    """(conversation_id, turn_id) composite key -> list of question_ids
    that cite it as evidence. Plain turn_id alone is NOT unique — dia_id
    format "D{session}:{turn}" repeats identically across every
    conversation, so this must be scoped per-conversation."""
    index: dict[str, list[str]] = defaultdict(list)
    for _, row in qa_df.iterrows():
        conv_id = row["conversation_id"]
        for tid in row["evidence_turn_ids"]:
            index[f"{conv_id}::{tid}"].append(row["question_id"])
    return dict(index)