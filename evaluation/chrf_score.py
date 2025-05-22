#!/usr/bin/env python3
"""
Evaluate Arabic-poem explanations – CHRF only
---------------------------------------------
* Aligns by poem id + verse index (≤ 10 verses in prediction)
* Arabic normalisation + CAMeL tokeniser
* Live progress bar with tqdm
"""

import json, re, unicodedata, argparse, statistics
from pathlib import Path
from typing   import Dict, List

# ── external libs ────────────────────────────────────────────────────────────
try:
    # clitic-aware segmenter (preferred)
    from camel_tools.tokenizers.word import simple_word_segment as _segment
except ImportError:
    # fallback: whitespace + punctuation
    from camel_tools.tokenizers.word import simple_word_tokenize as _segment
    print("[INFO] simple_word_segment not found – using simple_word_tokenize")

from evaluate import load
from tqdm import tqdm                               # ← progress-bar

CHRF = load("chrf")

# ── Arabic normalisation & tokeniser ─────────────────────────────────────────
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def norm(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def camel_tok(s: str) -> List[str]:
    return _segment(norm(s))

# ── helpers ──────────────────────────────────────────────────────────────────
def reconcile(pred: List[str], n_gt: int) -> List[str]:
    """Merge per-line explanations if necessary, keep ≤ n_gt items."""
    if n_gt < len(pred) <= 2 * n_gt:
        pred = [" ".join(pred[i:i + 2]) for i in range(0, len(pred), 2)]
    return pred[:n_gt]

def load_poems(path: Path, verse_key: str) -> Dict[str, List[str]]:
    poems = {}
    with path.open(encoding="utf-8") as fp:
        for poem in json.load(fp):
            pid = poem.get("id")
            if pid and verse_key in poem:
                poems[pid] = [norm(v["explanation"]) for v in poem[verse_key]]
    return poems

# ── per-poem CHRF ────────────────────────────────────────────────────────────
def chrf_poem(gt: List[str], pred: List[str]) -> Dict[str, float]:
    pred = reconcile(pred, len(gt))
    refs, hyps = [], []
    for k, ref in enumerate(gt):
        refs.append([ref])
        hyps.append(pred[k] if k < len(pred) else "")

    # guard zero tokens
    if all(len(camel_tok(h)) == 0 for h in hyps):
        chrf_score = 0.0
    else:
        chrf_score = CHRF.compute(predictions=hyps, references=refs)["score"] / 100

    cov = sum(bool(h) for h in hyps) / len(hyps)
    return {"chrf": chrf_score, "coverage": cov, "chrf_cov": chrf_score * cov}

# ── corpus driver ────────────────────────────────────────────────────────────
def evaluate_files(gt_path: Path, pred_path: Path):
    gt_poems   = load_poems(gt_path,   "explanation")
    pred_poems = load_poems(pred_path, "verse_explanations")

    shared = gt_poems.keys() & pred_poems.keys()
    if missing := pred_poems.keys() - shared:
        print(f"[WARN] {len(missing)} prediction ids have no GT – skipped.")
    if missing := gt_poems.keys() - shared:
        print(f"[WARN] {len(missing)} GT ids have no prediction – skipped.")

    agg = {"chrf": [], "coverage": [], "chrf_cov": []}

    for pid in tqdm(shared, desc="Scoring poems", unit="poem"):
        scores = chrf_poem(gt_poems[pid], pred_poems[pid])
        for k, v in scores.items():
            agg[k].append(v)

    return {k: statistics.mean(v) for k, v in agg.items()}

# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gt_json = "final_poems.json"

    modeljsons = {
        # "GPT-4o": "gpt_4o_generated_explanation.json",
        # "GPT-4o-mini": "gpt_4o_mini_generated_explanation.json",
        # "Gemini-1_5_pro": "gemini_1_5_pro_generated_explanation.json",
        # "Gemini-2_0_Flash": "gemini_2_0_flash_generated_explanation.json",
        # "gemini_2_5_flash": "gemini_2_5_flash_generated_explaneation.json",
        # "Llama_3_3_70B": "llama_3_3_70b_generated_explanation.json",
        # "Aya-Expanse": "aya_expanse_32b_generated_explanation.json",
        # "deepseek-v3": "deepseek_v3_generated_explanation.json",
        # "deepseek-R1": "deepseek_R1_generated_explanation.json",
        "qwen_3": "qwen_3_generated_explanation.json",
        "fanar_star": "fanar_star_generated_explanation.json",
    }

    for model, json_file in modeljsons.items():
        print(f"CHRF SCORES FOR {model}")
        results = evaluate_files(Path(gt_json), Path(json_file))

        print("\n=== Macro-averaged metrics (first ≤10 verses per poem) ===")
        for k, v in results.items():
            print(f"{k:10s}: {v:.4f}")

