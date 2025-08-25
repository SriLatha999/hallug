# methods/detector.py
import re

# Simple regex red flags commonly seen in made-up or overconfident answers
FLAG_PATTERNS = {
    "vague_study": re.compile(r"\b(studies?|research)\s+(show|suggest|indicate)\b", re.I),
    "according_to": re.compile(r"\baccording to\b", re.I),
    "fake_cite_hint": re.compile(r"\b(?:doi:|arxiv:\d|journal of|conference on)\b", re.I),
    "overconfident": re.compile(r"\b(definitely|certainly|undoubtedly|without doubt)\b", re.I),
    "year_soup": re.compile(r"\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b", re.I),  # multiple years
}

def pattern_flags(text: str):
    txt = (text or "")
    return [name for name, rx in FLAG_PATTERNS.items() if rx.search(txt)]

def shape_features(text: str):
    t = (text or "").strip()
    n = len(t)
    digits = sum(c.isdigit() for c in t)
    periods = t.count(".")
    return {
        "len": n,
        "digits_ratio": digits / max(1, n),
        "sentences": max(1, periods),
    }

def risk_score(question: str, answer: str):
    """Return (score in [0,1], flags list, features dict)."""
    flags = pattern_flags(answer)
    feats = shape_features(answer)
    score = 0.0
    # Each flag adds risk
    score += 0.18 * len(flags)
    # Too many digits (dates, numbers) → often spurious specifics
    if feats["digits_ratio"] > 0.12:
        score += 0.25
    # Very long answers are often padded speculation
    if feats["len"] > 380:
        score += 0.20
    return max(0.0, min(1.0, score)), flags, feats
