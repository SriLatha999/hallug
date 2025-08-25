# run_entropy_test.py
import argparse, json
from pathlib import Path

from data_utils import load_truthfulqa, load_or_bench, load_splits
from methods.prompt_control import apply_policy
from methods.entropy_gate import entropy_gate_generate

from baseline import (
    MODELS, build_backend, cached_generate, RUN_TAG,
    refusal_flag, includes_any, golds_from_tqa, GEN
)

OUT = Path("results"); OUT.mkdir(exist_ok=True)

def get_chosen_pc_level(model_key: str):
    p = OUT / f"prompt_dev_{model_key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())["chosen_level"]
        except Exception:
            return None
    return None

def get_chosen_tau(model_key: str):
    p = OUT / f"entropy_dev_{model_key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())["chosen_tau"], json.loads(p.read_text()).get("k", 4)
        except Exception:
            return None, 4
    return None, 4

def run_test(model_key: str):
    model_id = MODELS[model_key]
    backend, mid_disp = build_backend(model_id)

    splits = load_splits()
    tqa_idx = splits["tqa"]["test"]
    ben_idx = splits["ben"]["test"]
    tox_idx = splits["tox"]["test"]

    pc_level = get_chosen_pc_level(model_key)
    tau_bits, k = get_chosen_tau(model_key)

    # --- TruthfulQA ---
    tqa = load_truthfulqa()
    tqa_hits = 0
    for i in tqa_idx:
        q = tqa[i]["question"]
        base = q
        prompt = apply_policy(base, pc_level) if pc_level else f"Answer concisely (1–2 sentences):\n{base}"
        ans, H, _clusters = entropy_gate_generate(backend, f"{mid_disp}:eg:{RUN_TAG}", prompt, k=k, tau_bits=tau_bits)
        if H >= tau_bits:
            ans = "I cannot answer confidently; I may be hallucinating."
        if includes_any(ans, golds_from_tqa(tqa[i])):
            tqa_hits += 1
    tqa_acc = tqa_hits / len(tqa_idx) if tqa_idx else 0.0

    # --- OR-Bench ---
    ben, tox = load_or_bench()

    ben_flags = 0
    for i in ben_idx:
        base = ben[i]["prompt"]
        prompt = apply_policy(base, pc_level) if pc_level else base
        ans, H, _ = entropy_gate_generate(backend, f"{mid_disp}:eg:{RUN_TAG}", prompt, k=k, tau_bits=tau_bits)
        if H >= tau_bits:
            ans = "I cannot answer confidently; I may be hallucinating."
        if refusal_flag(ans):
            ben_flags += 1
    ben_over = ben_flags / len(ben_idx) if ben_idx else 0.0

    tox_flags = 0
    for i in tox_idx:
        base = tox[i]["prompt"]
        prompt = apply_policy(base, pc_level) if pc_level else base
        ans, H, _ = entropy_gate_generate(backend, f"{mid_disp}:eg:{RUN_TAG}", prompt, k=k, tau_bits=tau_bits)
        if H >= tau_bits:
            ans = "I cannot answer confidently; I may be hallucinating."
        if refusal_flag(ans):
            tox_flags += 1
    tox_refusal = tox_flags / len(tox_idx) if tox_idx else 0.0

    summary = {
        "model_key": model_key,
        "model_id": mid_disp,
        "pc_level": pc_level,
        "chosen_tau": tau_bits,
        "k": k,
        "test_metrics": {
            "tqa_acc": tqa_acc,
            "ben_over": ben_over,
            "tox_refusal": tox_refusal
        }
    }
    (OUT / f"entropy_test_{model_key}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), required=True)
    args = ap.parse_args()
    run_test(args.model)
