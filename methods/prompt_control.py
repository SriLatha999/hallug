POLICY = {
    "low": (
        "Rules:\n"
        "- If you know the answer, state it directly.\n"
        "- If uncertain, say exactly: \"I'm not fully sure.\"\n"
        "- Keep answers to 1–2 sentences.\n"
        "- Avoid speculation."
    ),
    "medium": (
        "Rules:\n"
        "- If you know the answer, state it directly.\n"
        "- If uncertain, say exactly: \"I'm not fully sure.\"\n"
        "- State only verifiable facts; avoid guessing dates/names.\n"
        "- Keep answers to 1–2 sentences."
    ),
    "high": (
        "Rules (strict):\n"
        "- If you know the answer, state it directly.\n"
        "- If not confident, say exactly: \"I'm not fully sure\" and stop.\n"
        "- Do not guess numbers/dates/names.\n"
        "- Keep answers to 1–2 sentences."
        "- Provide only verified parts you’re confident about; avoid refusing benign requests- Do not mention policies; just answer cautiously."
    ),
}

def apply_policy(question: str, level: str) -> str:
    return f"{POLICY[level]}\n\nQuestion:\n{question}\n\nAnswer:"
