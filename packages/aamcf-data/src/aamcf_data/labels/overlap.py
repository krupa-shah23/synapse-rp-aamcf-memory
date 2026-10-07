"""Shared label-construction utilities across LoCoMo and LongMemEval.

label_overlap reimplements LRE's annotation-free proxy exactly as in the LRE
repo (experiments/conversational_qa/proxy_labels.py, normalize in dataset.py):
a unit (turn or session) is positive if its tokens cover >= tau of some
answer's tokens. The fraction is answer-side recall, |answer ∩ unit| / |answer|,
not the share of the unit found in the answer. tau=0.4 is LRE's threshold
(flat-optimal in [0.3, 0.5] per their joint sweep across both datasets).
"""

from __future__ import annotations

import re

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """LRE's normalization: lowercase, drop articles, drop punctuation."""
    text = _ARTICLES.sub(" ", text.lower())
    text = _PUNCT.sub("", text)
    return " ".join(text.split())


def content_tokens(text: str) -> set[str]:
    """Token set after LRE's normalization. Exact lexical match, no stopword
    list beyond articles — intentionally the cheap/annotation-free proxy."""
    return set(normalize(text or "").split())


def label_overlap(unit_text: str, answer_texts: list[str], tau: float = 0.4) -> int:
    """1 if the unit's tokens cover >= tau of ANY answer's tokens (answer-side
    recall), else 0. Empty answers are skipped, as in LRE."""
    unit_toks = content_tokens(unit_text)
    best = 0.0
    for a in answer_texts:
        if not a or not a.strip():
            continue
        ans_toks = content_tokens(a)
        if not ans_toks:
            continue
        inter = len(unit_toks & ans_toks)
        if inter:
            best = max(best, inter / len(ans_toks))
    return int(best >= tau)
