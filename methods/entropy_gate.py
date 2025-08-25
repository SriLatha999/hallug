import math, re
from collections import Counter
from baseline import cached_generate, GEN

_CLEAN = re.compile(r"[^a-z0-9 .,:;'\"!?-]")

def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _CLEAN.sub("", s)
    # take only the first sentence-ish to reduce surface variance
    for sep in [".", "?", "!"]:
        if sep in s:
            s = s.split(sep)[0]
            break
    return " ".join(s.split())

def entropy_from_counts(counts):
    total = sum(counts.values()) or 1
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0: H -= p * math.log2(p)
    return H

def entropy_gate_generate(backend, model_id: str, prompt: str, k: int = 4, tau_bits: float = 1.0):
    GEN_SAMP = dict(max_new_tokens=GEN.get("max_new_tokens", 64), temperature=0.6, top_p=0.9)

    samples = []
    normed = []
    for i in range(k):
        # cache key segment includes ":k{i}" so each sample is distinct
        rec = cached_generate(backend, f"{model_id}:k{i}", prompt, **GEN_SAMP)
        txt = rec["text"]
        samples.append(txt)
        normed.append(_normalize_text(txt))

    counts = Counter(normed)
    H = entropy_from_counts(counts)

    if H <= tau_bits:
        # choose the dominant cluster's *first* original sample, to preserve formatting
        choice_norm, _ = counts.most_common(1)[0]
        for s in samples:
            if _normalize_text(s) == choice_norm:
                return s, H, counts
    # abstain
    return "I'm not fully sure.", H, counts
