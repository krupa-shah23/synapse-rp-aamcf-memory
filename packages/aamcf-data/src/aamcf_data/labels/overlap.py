"""Shared label-construction utilities across LoCoMo and LongMemEval.

label_overlap = LRE's annotation-free proxy: a unit (turn or session) is
positive if it shares >= tau content-token overlap with the eventual
answer text. tau=0.4 is LRE's validated threshold (flat-optimal in [0.3, 0.5]
per their joint sweep across both datasets).
"""

from __future__ import annotations

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "and", "or", "but", "if", "so", "because", "as", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "out", "on", "off", "over", "under",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "this", "that", "these", "those", "there", "here",
    "not", "no", "yes", "okay", "ok", "sure", "yeah", "um", "uh",
})


def content_tokens(text: str) -> set[str]:
    """Lowercase content-word tokens, stopwords stripped. Exact lexical
    overlap, not semantic — this is intentionally the cheap/annotation-free
    proxy, not a similarity model."""
    if not text:
        return set()
    raw = text.lower().split()
    cleaned = {tok.strip(".,!?;:\"'()[]") for tok in raw}
    return {tok for tok in cleaned if tok and tok not in STOPWORDS}


def label_overlap(unit_text: str, answer_texts: list[str], tau: float = 0.4) -> int:
    """1 if unit_text shares >= tau fraction of its content tokens with
    ANY of the given answers, else 0."""
    unit_toks = content_tokens(unit_text)
    if not unit_toks or not answer_texts:
        return 0
    best = max(
        (len(unit_toks & content_tokens(a)) / len(unit_toks) for a in answer_texts),
        default=0.0,
    )
    return int(best >= tau)