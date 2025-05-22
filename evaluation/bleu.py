# #!/usr/bin/env python3
# """
# eval_poem_explanations.py
# -------------------------
#  Evaluate LLM-generated Arabic-poem explanations when
#  • GT file  = list[{"id", "explanation", "raw_explanation", ...}]
#               └─  explanation = list[{verse, explanation}]
#  • Pred file= list[{"id", "verse_explanations", "complete_explanation", ...}]
#               └─  verse_explanations = list[{verse, explanation}]
#               └─  first ≤ 10 verses only

# Outputs macro-averaged BLEU×Coverage, chrF++, BERTScore-F1, Coverage %.
# """

# import json, re, unicodedata, math, statistics, argparse
# from pathlib import Path
# from typing   import Dict, List

# # ── external libs ─────────────────────────────────────────────────────────────
# try:
#     # preferred: splits proclitics + enclitics
#     from camel_tools.tokenizers.word import simple_word_segment as _segment
# except ImportError:
#     # fallback: whitespace+punctuation only
#     from camel_tools.tokenizers.word import simple_word_tokenize as _segment
#     print("[INFO] simple_word_segment not found – using simple_word_tokenize")

# from evaluate import load

# BLEU     = load("bleu")
# CHRF     = load("chrf")          # chrF++ if 'word_order=2' (default)
# # BERTSCR  = load("bertscore")

# # ── arabic normalisation & tokeniser ──────────────────────────────────────────
# _DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

# def norm(text: str) -> str:
#     text = _DIACRITICS.sub("", text)             # remove tashkīl
#     text = unicodedata.normalize("NFKC", text)   # canonical form
#     text = re.sub(r"\s+", " ", text)             # squeeze WS
#     return text.strip()

# def camel_tok(s: str) -> List[str]:
#     """normalise + segment into clitic-aware tokens"""
#     return _segment(norm(s))

# # ── helpers ───────────────────────────────────────────────────────────────────
# def reconcile(pred:list, n_gt:int)->List[str]:
#     """
#     If model produced per-line explanations (≈2× longer than verses),
#     merge every 2 lines; then truncate/pad to length n_gt.
#     """
#     if n_gt < len(pred) <= 2 * n_gt:
#         merged = [" ".join(pred[i:i+2]) for i in range(0, len(pred), 2)]
#         pred = merged
#     return pred[:n_gt]

# def load_poems(path:Path, verse_key:str)->Dict[str, List[str]]:
#     """
#     Return {poem_id → list[str] explanations_in_order}.
#     verse_key = "explanation" or "verse_explanations".
#     """
#     poems = {}
#     with path.open(encoding="utf-8") as fp:
#         for poem in json.load(fp):
#             pid = poem.get("id")
#             if pid and verse_key in poem:
#                 poems[pid] = [norm(v["explanation"])
#                               for v in poem[verse_key]]
#     return poems

# # ── per-poem evaluation ───────────────────────────────────────────────────────
# def eval_poem(gt:List[str], pred:List[str]) -> Dict[str,float]:
#     pred = reconcile(pred, len(gt))
#     hyps, refs = [], []
#     for k, ref in enumerate(gt):
#         refs.append([ref])
#         hyps.append(pred[k] if k < len(pred) else "")   # "" if missing

#     bleu = BLEU.compute(predictions=hyps, references=refs,
#                         tokenizer=camel_tok, smooth=True)["bleu"]
#     chrf = CHRF.compute(predictions=hyps, references=refs)["score"] / 100
#     # bert = BERTSCR.compute(predictions=hyps, references=[r[0] for r in refs],
#     #                        lang="ar")["f1"]
#     # bert = statistics.mean(bert)
#     coverage = sum(bool(h) for h in hyps) / len(hyps)
#     return {"bleu": bleu,
#             "chrF": chrf,
#             # "berts": bert,
#             "coverage": coverage,
#             "bleu_cov": bleu * coverage}

# # ── corpus driver ─────────────────────────────────────────────────────────────
# def evaluate_files(gt_path:Path, pred_path:Path):
#     gt_poems   = load_poems(gt_path,   "explanation")
#     pred_poems = load_poems(pred_path, "verse_explanations")

#     shared = gt_poems.keys() & pred_poems.keys()
#     if missing := pred_poems.keys() - shared:
#         print(f"[WARN] {len(missing)} prediction ids have no GT – skipped.")
#     if missing := gt_poems.keys() - shared:
#         print(f"[WARN] {len(missing)} GT ids have no prediction – skipped.")

#     macro = {"bleu": [], "chrF": [], "berts": [], "coverage": [], "bleu_cov": []}

#     for pid in shared:
#         scores = eval_poem(gt_poems[pid], pred_poems[pid])
#         for k,v in scores.items(): macro[k].append(v)

#     report = {k: statistics.mean(v) for k,v in macro.items()}
#     return report

# # ── CLI ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     gt_json = "final_poems.json"
#     pred_json = "gpt_4o_generated_explanation.json"

#     res = evaluate_files(Path(gt_json), Path(pred_json))
#     print("\n=== Macro-averaged metrics (first ≤10 verses per poem) ===")
#     for k,v in res.items():
#         print(f"{k:10s}: {v:.4f}")







#!/usr/bin/env python3
"""
Evaluate Arabic-poem explanations – BLEU only
--------------------------------------------
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

BLEU = load("bleu")

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

# ── per-poem BLEU ────────────────────────────────────────────────────────────
def bleu_poem(gt: List[str], pred: List[str]) -> Dict[str, float]:
    pred = reconcile(pred, len(gt))
    refs, hyps = [], []
    for k, ref in enumerate(gt):
        refs.append([ref])
        hyps.append(pred[k] if k < len(pred) else "")

    # --- NEW: skip BLEU computation if model produced zero tokens -------------
    if all(len(camel_tok(h)) == 0 for h in hyps):   # ← guard
        bleu = 0.0
    else:
        bleu = BLEU.compute(predictions=hyps, references=refs,
                            tokenizer=camel_tok, smooth=True)["bleu"]
    # -------------------------------------------------------------------------

    cov  = sum(bool(h) for h in hyps) / len(hyps)
    return {"bleu": bleu, "coverage": cov, "bleu_cov": bleu * cov}

# ── corpus driver ────────────────────────────────────────────────────────────
def evaluate_files(gt_path: Path, pred_path: Path):
    gt_poems   = load_poems(gt_path,   "explanation")
    pred_poems = load_poems(pred_path, "verse_explanations")

    shared = gt_poems.keys() & pred_poems.keys()
    if missing := pred_poems.keys() - shared:
        print(f"[WARN] {len(missing)} prediction ids have no GT – skipped.")
    if missing := gt_poems.keys() - shared:
        print(f"[WARN] {len(missing)} GT ids have no prediction – skipped.")

    agg = {"bleu": [], "coverage": [], "bleu_cov": []}

    for pid in tqdm(shared, desc="Scoring poems", unit="poem"):
        scores = bleu_poem(gt_poems[pid], pred_poems[pid])
        for k, v in scores.items():
            agg[k].append(v)

    return {k: statistics.mean(v) for k, v in agg.items()}

# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GPT-4o BLEU Score")
    gt_json = "final_poems.json"
    pred_json = "gpt_4o_generated_explanation.json"

    results = evaluate_files(Path(gt_json), Path(pred_json))

    print("\n=== Macro-averaged metrics (first ≤10 verses per poem) ===")
    for k, v in results.items():
        print(f"{k:10s}: {v:.4f}")

