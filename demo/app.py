import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # .../hallug
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, random, re
import streamlit as st
from dotenv import load_dotenv

# our modules
from data_utils import load_truthfulqa
from baseline import (
    MODELS, build_backend, cached_generate, RUN_TAG, GEN,
    includes_any, golds_from_tqa, refusal_flag
)
from methods.prompt_control import apply_policy
from methods.entropy_gate import entropy_gate_generate

load_dotenv()
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

# ---------------------------
# Helpers to read tuned config
# ---------------------------
def _read_json_field(path: Path, key: str, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text()).get(key, default)
    except Exception:
        pass
    return default

def chosen_pc_level(model_key: str) -> str:
    return _read_json_field(RESULTS / f"prompt_dev_{model_key}.json", "chosen_level", "medium")

def chosen_tau_and_k(model_key: str):
    p = RESULTS / f"entropy_dev_{model_key}.json"
    tau = _read_json_field(p, "chosen_tau", 1.0)
    k   = _read_json_field(p, "k", 4)
    return float(tau), int(k)

# ---------------------------
# Very light risk heuristic
# ---------------------------
SUS_PHRASES = [
    r"\b(studies? (say|show))\b", r"\b(according to (research|sources|experts))\b",
    r"\b(as everyone knows)\b", r"\b(it is well known)\b",
]
HEDGE = [r"\bmaybe\b", r"\bprobably\b", r"\bI think\b", r"\bI guess\b", r"\bnot sure\b"]
NUM_FAB = [r"\b\d{4}\b", r"\b\d+(\.\d+)?%?\b"]  # bare years/numbers

def risk_score(question: str, answer: str):
    text = (answer or "").strip()
    q = (question or "").strip()
    flags = set()

    for pat in SUS_PHRASES:
        if re.search(pat, text, re.I):
            flags.add("generic-claims")

    if re.search(NUM_FAB[0], text) or re.search(NUM_FAB[1], text):
        if not re.search(r"(citation|ref|paper|dataset|source)", text, re.I):
            flags.add("bare-numbers")

    hedges = sum(bool(re.search(h, text, re.I)) for h in HEDGE)

    L, qL = len(text.split()), len(q.split())
    if L > max(40, 2.5 * qL): flags.add("ramble")
    if L < 4: flags.add("too-short")
    if refusal_flag(text): flags.add("refusal")

    base = 0.15 * len(flags) - 0.05 * hedges
    base = max(0.0, min(1.0, base))
    return base, sorted(flags)

# ---------------------------
# UI setup & styles
# ---------------------------
st.set_page_config(page_title="Layered Hallucination Guard — Flow", layout="wide")
st.markdown("""
<style>
.card {border:1px solid #E6EAF2; border-radius:14px; padding:14px; background:#fff;}
.card h4 {margin:0 0 8px 0;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; background:#EEF2FF; color:#3949ab; font-size:0.75rem; margin-right:8px;}
.arrow {text-align:center; font-size:26px; color:#A0AEC0; padding-top:46px;}
.small {color:#667085; font-size:0.9rem;}
.codebox {background:#0f172a; color:#e2e8f0; padding:10px 12px; border-radius:8px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas;}
header, .css-18ni7ap {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Layered Hallucination Guard — Flow Demo")
st.caption("Baseline → Prevention (Prompt Control) → Detection (Risk) → Verification (Entropy Gate) → Final Output")

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")
    model_key = st.selectbox("Model", list(MODELS.keys()), index=0)
    pc_default = chosen_pc_level(model_key)
    tau_default, k_default = chosen_tau_and_k(model_key)

    pc_level = st.selectbox("Prompt Control", ["low","medium","high"],
                            index=["low","medium","high"].index(pc_default))
    risk_thr = st.slider("Risk threshold (Detection)", 0.0, 1.0, 0.50, 0.05)
    tau_bits = st.select_slider("Entropy τ (bits)", options=[0.6,0.8,1.0,1.2,1.5], value=float(tau_default))
    k = st.slider("k samples (Entropy Gate)", 3, 8, int(k_default), 1)

    st.divider()
    st.subheader("TruthfulQA helper")
    if st.button("Pick random TruthfulQA question"):
        ds = load_truthfulqa()
        i = random.randrange(len(ds))
        st.session_state["prompt_text"] = ds[i]["question"]
        st.session_state["tqa_gold"] = golds_from_tqa(ds[i])

# ---------------------------
# Input row
# ---------------------------
default_q = st.session_state.get("prompt_text", "Who discovered penicillin?")
col_q, col_run = st.columns([0.8, 0.2])
with col_q:
    user_q = st.text_input("Your question", default_q, key="prompt_text_box")
with col_run:
    st.write("")
    run = st.button("▶ Run Pipeline", use_container_width=True)

# backend
backend, mid_disp = build_backend(MODELS[model_key])

# ---------------------------
# The single-row flow
# ---------------------------
c_base, c_a1, c_pc, c_a2, c_risk, c_a3, c_gate, c_a4, c_final = st.columns([1.4,0.13,1.6,0.13,1.2,0.13,1.6,0.13,1.6])

# Step 0: Baseline
with c_base:
    st.markdown('<div class="card"><h4>Baseline (no PC)</h4>', unsafe_allow_html=True)
    if run:
        base_prompt = f"Answer concisely (1–2 sentences):\n{user_q}"
        rec = cached_generate(backend, f"{mid_disp}:baseline:{RUN_TAG}", base_prompt, **GEN)
        base_text = rec["text"]
        st.markdown(f'<div class="small"><b>Output</b></div><div class="codebox">{base_text}</div>', unsafe_allow_html=True)
    else:
        base_text = ""
        st.info("Click ▶ Run Pipeline")
    st.markdown('</div>', unsafe_allow_html=True)

with c_a1: st.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)

# Step 1: Prompt Control
with c_pc:
    st.markdown(f'<div class="card"><h4>Layer 1 — Prevention (Prompt Control)</h4><span class="badge">PC: {pc_level}</span>', unsafe_allow_html=True)
    if run:
        pc_prompt = apply_policy(user_q, pc_level)
        rec_pc = cached_generate(backend, f"{mid_disp}:pc:{pc_level}:{RUN_TAG}", pc_prompt, **GEN)
        pc_text = rec_pc["text"]

        with st.expander("Show PC prompt prefix", expanded=False):
            st.markdown(f'<div class="codebox">{pc_prompt}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="small"><b>Output</b></div><div class="codebox">{pc_text}</div>', unsafe_allow_html=True)
    else:
        pc_text = ""
    st.markdown('</div>', unsafe_allow_html=True)

with c_a2: st.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)

# Step 2: Detection (Risk)
with c_risk:
    st.markdown('<div class="card"><h4>Layer 2 — Detection (Risk)</h4>', unsafe_allow_html=True)
    if run:
        r, flags = risk_score(user_q, pc_text)
        st.progress(min(1.0, r), text=f"Risk: {r:.2f}  (threshold = {risk_thr:.2f})")
        st.markdown("**Flags:** " + (", ".join(flags) if flags else "_none_"))
        gated = r >= risk_thr
        st.markdown(f"**Gate?** {'Yes' if gated else 'No'}")
    else:
        gated = False
    st.markdown('</div>', unsafe_allow_html=True)

with c_a3: st.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)

# Step 3: Verification (Entropy Gate)
with c_gate:
    st.markdown('<div class="card"><h4>Layer 3 — Verification (Entropy Gate)</h4>', unsafe_allow_html=True)
    if run and gated:
        pc_prompt = apply_policy(user_q, pc_level)  # same input used for tuning
        ans, H, clusters = entropy_gate_generate(
            backend, f"{mid_disp}:eg:{RUN_TAG}", pc_prompt, k=int(k), tau_bits=float(tau_bits)
        )
        triggered = H >= float(tau_bits)
        st.markdown(f"**Entropy** H = {H:.2f} bits (τ = {tau_bits:.2f}) → **Triggered?** {'Yes' if triggered else 'No'}")
        with st.expander("Show clusters/samples"):
            if clusters:
                for i, item in enumerate(clusters):
                    try: rep, count = item
                    except: rep, count = str(item), None
                    st.markdown(f"- Cluster {i+1} (n={count}): {rep}")
            st.markdown(f'<div class="small"><b>Gate candidate</b></div><div class="codebox">{ans}</div>', unsafe_allow_html=True)
        final_text = "ABSTAIN: I cannot answer confidently; I may be hallucinating." if triggered else ans
    elif run and not gated:
        st.caption("Risk below threshold → Entropy Gate not invoked.")
        final_text = pc_text
    else:
        final_text = ""
    st.markdown('</div>', unsafe_allow_html=True)

with c_a4: st.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)

# Final
with c_final:
    st.markdown('<div class="card"><h4>Final Output</h4>', unsafe_allow_html=True)
    if run:
        st.markdown(f'<div class="codebox">{final_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Optional: quick gold check
with st.expander("Evaluate against TruthfulQA gold (if available)"):
    gold = st.session_state.get("tqa_gold")
    if gold and run:
        ok = includes_any(final_text, gold)
        st.write("Gold answers:", gold)
        st.success("✔ Matches a gold answer") if ok else st.warning("✘ Does not match a gold answer")
    elif not gold:
        st.caption("Pick a TruthfulQA question in the sidebar to enable this check.")
