from pathlib import Path
import json, random
from datasets import load_dataset

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)

def load_truthfulqa():
    return load_dataset("truthful_qa", "generation")["validation"]

def load_or_bench():
    ben = load_dataset("bench-llm/or-bench", name="or-bench-80k")["train"]   # benign
    tox = load_dataset("bench-llm/or-bench", name="or-bench-toxic")["train"] # toxic
    return ben, tox


def make_splits(n_dev=30, n_test=150, seed=7):
    random.seed(seed)
    tqa = load_truthfulqa(); ben, tox = load_or_bench()

    def pick(n_total):
        idx = list(range(n_total)); random.shuffle(idx)
        return {"dev": idx[:n_dev], "test": idx[n_dev:n_dev+n_test]}

    splits = {"tqa": pick(len(tqa)), "ben": pick(len(ben)), "tox": pick(len(tox))}
    (DATA_DIR / "splits.json").write_text(json.dumps(splits, indent=2))
    return splits

def load_splits():
    p = DATA_DIR / "splits.json"
    if p.exists():
        import json
        return json.loads(p.read_text())
    from data_utils import make_splits
    return make_splits()

