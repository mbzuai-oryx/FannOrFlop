#!/usr/bin/env python3
"""
eval_bertscore.py
-----------------
Compute BERTScore (P/R/F1) for LLM-generated Arabic-poem explanations,
using AraBERT v2 as the underlying model.

Ground-truth JSON  : list[{id, explanation=list[{..., explanation}]}]
Prediction JSON    : list[{id, verse_explanations=list[{..., explanation}]}]
Each prediction keeps only the first ≤ 10 verses.

Macro-average across poems.
"""

import json
import re
import unicodedata
import statistics
import argparse
from pathlib import Path
from typing import Dict, List

# ── external libs ────────────────────────────────────────────────────────────
try:
    from camel_tools.tokenizers.word import simple_word_segment as _segment
except ImportError:
    from camel_tools.tokenizers.word import simple_word_tokenize as _segment

from evaluate import load
from tqdm import tqdm

BS = load("bertscore")

# ── Arabic normalisation & tokeniser ─────────────────────────────────────────
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def norm(text: str) -> str:
    # strip diacritics, normalize forms, remove extra spaces
    text = _DIACRITICS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ـ", "")                       # remove tatweel
    text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~،؟«»]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(_segment(text))

# ── helpers ──────────────────────────────────────────────────────────────────
def reconcile(pred: List[str], limit: int) -> List[str]:
    """Merge per-line explanations if len≈2×limit, then clip to limit."""
    if limit < len(pred) <= 2 * limit:
        pred = [" ".join(pred[i:i+2]) for i in range(0, len(pred), 2)]
    return pred[:limit]

def load_poems(path: Path, key: str) -> Dict[str, List[str]]:
    poems = {}
    with path.open(encoding="utf-8") as fp:
        for poem in json.load(fp):
            pid = poem.get("id")
            if pid and key in poem:
                poems[pid] = [v["explanation"] for v in poem[key]]
    return poems

def evaluate_files(gt_path: Path, pr_path: Path):
    # load
    gt = load_poems(gt_path, "explanation")
    pr = load_poems(pr_path, "verse_explanations")

    shared = list(gt.keys() & pr.keys())
    if miss := pr.keys() - set(shared):
        print(f"[WARN] {len(miss)} prediction ids have no GT – skipped.")
    if miss := gt.keys() - set(shared):
        print(f"[WARN] {len(miss)} GT ids have no prediction – skipped.")

    preds, refs = [], []

    for pid in tqdm(shared, desc="Collecting pairs", unit="poem"):
        gt_verses = gt[pid][:10]
        pr_verses = reconcile(pr[pid], len(gt_verses))

        # concat + normalize
        ref = " ".join(norm(v) for v in gt_verses)
        hyp = " ".join(norm(v) for v in pr_verses)

        refs.append(ref)
        preds.append(hyp)

    # compute BERTScore over the entire corpus
    results = BS.compute(
        predictions=preds,
        references=refs,
        lang="ar",
        model_type="aubmindlab/bert-base-arabertv02",
        num_layers=12,
        batch_size=16,
    )

    # macro-average
    p_mean = statistics.mean(results["precision"])
    r_mean = statistics.mean(results["recall"])
    f_mean = statistics.mean(results["f1"])

    return {"precision": p_mean, "recall": r_mean, "f1": f_mean}

# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    modeljsons = {
        # "GPT-4o": "gpt_4o_generated_explanation.json",
        # "GPT-4o-mini": "gpt_4o_mini_generated_explanation.json",
        # "Gemini-1_5_pro": "gemini_1_5_pro_generated_explanation.json",
        # "Gemini-2_0_Flash": "gemini_2_0_flash_generated_explanation.json",
        # "Llama_3_3_70B": "llama_3_3_70b_generated_explanation.json",
        # "Aya-Expanse": "aya_expanse_32b_generated_explanation.json",
        "deepseek-v3": "deepseek_v3_generated_explanation.json",
        "deepseek-R1": "deepseek_R1_generated_explanation.json",
        # "gemini_2_5_flash": "gemini_2_5_flash_generated_explanation.json"
    }

    gt_json = "final_poems.json"

    for model, json_file in modeljsons.items():
        print(f"{model} BERTSCORE")
        res = evaluate_files(Path(gt_json), Path(json_file))
        print("\n=== Macro-averaged BERTScore (AraBERT) ===")
        print(f"Precision : {res['precision']:.4f}")
        print(f"Recall    : {res['recall']:.4f}")
        print(f"F1-score  : {res['f1']:.4f}")
        print("="*10)
