# run_prompt_test.py
import argparse, json
from pathlib import Path

from data_utils import load_truthfulqa, load_or_bench, load_splits
from methods.prompt_control import apply_policy
from baseline import (
    MODELS, GEN, build_backend, cached_generate,
    refusal_flag, includes_any, golds_from_tqa, SAMPLE_N
)

OUT = Path("results"); OUT.mkdir(exist_ok=True)

def slice_idx(idx):
    if SAMPLE_N is None: return idx
    return idx[:SAMPLE_N]

def main(model_key: str):
    # load chosen level from dev run
    dev_path = OUT / f"prompt_dev_{model_key}.json"
    if not dev_path.exists():
        raise SystemExit("Run run_prompt_dev.py first for this model.")
    dev = json.loads(dev_path.read_text())
    level = dev["chosen_level"]

    model_id = MODELS[model_key]
    backend, mid_disp = build_backend(model_id)
    splits = load_splits()
    tqa_idx = slice_idx(splits["tqa"]["test"])
    ben_idx = slice_idx(splits["ben"]["test"])
    tox_idx = slice_idx(splits["tox"]["test"])

    # datasets
    tqa = load_truthfulqa()
    ben, tox = load_or_bench()

    # ---- TruthfulQA acc ----
    hits = 0
    for i in tqa_idx:
        prompt = apply_policy(tqa[i]["question"], level)
        txt = cached_generate(backend, f"{mid_disp}:test:pc:{level}", prompt, **GEN)["text"]
        if includes_any(txt, golds_from_tqa(tqa[i])): hits += 1
    tqa_acc = hits / len(tqa_idx) if tqa_idx else 0.0

    # ---- Benign over-refusal ----
    flags = 0
    for i in ben_idx:
        prompt = apply_policy(ben[i]["prompt"], level)
        txt = cached_generate(backend, f"{mid_disp}:test:pc:{level}", prompt, **GEN)["text"]
        if refusal_flag(txt): flags += 1
    ben_over = flags / len(ben_idx) if ben_idx else 0.0

    # ---- Toxic refusal ----
    tox_flags = 0
    for i in tox_idx:
        prompt = apply_policy(tox[i]["prompt"], level)
        txt = cached_generate(backend, f"{mid_disp}:test:pc:{level}", prompt, **GEN)["text"]
        if refusal_flag(txt): tox_flags += 1
    tox_refusal = tox_flags / len(tox_idx) if tox_idx else 0.0

    out = {
        "model_key": model_key,
        "model_id": mid_disp,
        "chosen_level": level,
        "test_metrics": {
            "tqa_acc": tqa_acc,
            "ben_over": ben_over,
            "tox_refusal": tox_refusal
        }
    }
    (OUT / f"prompt_test_{model_key}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dense","moe"], default="dense")
    args = ap.parse_args()
    main(args.model)
