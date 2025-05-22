#!/usr/bin/env python3
"""
poem_judge_eval.py
------------------
Evaluates AI-generated poem explanations at the poem level.
Produces entailment, faithfulness, fluency, and overall scores per poem.
"""

import json, os, logging, threading, unicodedata, re, time, sys
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
import os
import sys


# ─── Configuration ────────────────────────────────────────────────────────
GT_FILE      = "final_poems.json"
PRED_FILE    = "fanar_star_generated_explanation.json"
OUT_FILE     = "judge_results/fanar_star-results.json"
MAX_WORKERS  = 10        # adjust to quota; 1-3 if rate-limited
REQUEST_PAUSE = 0.4     # stagger threads (seconds)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("CRITICAL: OPENAI_API_KEY environment variable not set. Please set it before running the script.")
    sys.exit(1)
client = OpenAI(api_key=api_key)


class Evaluations_Scores(BaseModel):
    faithfulness_score: float
    fluency_score: float
    overall_score: float


# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)


SYSTEM_PROMPT = """You are an expert Arabic linguist and literary evaluator.

Your task is to **evaluate a full Arabic poem’s verse-by-verse explanations**. You will compare **ground-truth** (human-written) explanations with **generated** explanations from an AI model.

You will judge each verse explanation based on the following two criteria:

---

### Evaluation Criteria (per verse)

1. **Faithfulness / Consistency**:  
   Is the generated explanation consistent with the meaning of the verse?  
   - Score 5: Deeply faithful to the verse’s content  
   - Score 3: General alignment but loses poetic imagery  
   - Score 1: Misinterprets or invents meaning

2. **Fluency / Grammaticality**:  
   Is the generated explanation well-formed Modern Standard Arabic?  
   - Score 5: Fluent, grammatically correct  
   - Score 3: Understandable with minor issues  
   - Score 1: Awkward, incomplete, or ungrammatical

---

### What You Will Receive

You will receive for each poem:
- `poem_title`
- "ground_truth": a list of objects { "v": <int>, "text": <string> }
- "generated"   : a list of objects with the **same v indices**

---

### What You Must Do

- “Compare all verses together and assign a single score 1–5 for each criterion.  
- Do **not** provide per-verse scores or any comments.”

Then:
- Calculate average scores for the whole poem
- Provide an `overall_score` (1–5) that reflects your judgment across all verses

---

Do NOT provide any comments or rationale.
Respond with valid JSON **only** in this format:

### Output Format (in JSON)

{
  "faithfulness_score": <1–5>,
  "fluency_score": <1–5>,
  "overall_score": <1–5>
}
"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

import re, unicodedata

def norm(text: str) -> str:
    """
    Minimal normaliser:
      • NFKC unicode normalisation
      • collapse all whitespace (incl. newlines) to single spaces
      • strip common Arabic diacritics
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)                     # collapse \n, \t, doubles
    text = re.sub(r"[\u064B-\u065F\u0610-\u061A"
                  r"\u06D6-\u06ED]", "", text)           # strip Arabic harakat
    return text.strip()


# ─── Helpers ───────────────────────────────────────────────────────────────
def load(path: str) -> List[Dict[str,Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_atomic(data: List[Dict[str,Any]], path: str):
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, path)

def build_pred_lookup(pred) -> Dict[str, List[Dict[str,str]]]:
    return {s["id"]: s.get("verse_explanations", []) for s in pred}

def evaluate_poem(poem):
    """
    Send one chat-completion per poem.
    Returns the parsed JSON dict with scores.
    """
    user_payload = {
        "id": poem["id"],
        "title": poem["title"],
        "ground_truth": poem["ground_truth"],
        "generated": poem["generated"]
    }

    response = client.responses.parse(
        model='gpt-4o-2024-08-06',
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        temperature=0.0,
        text_format=Evaluations_Scores,
    )

    content = response.output_parsed

    try:
        result = {
            "id": poem["id"],
            "title": poem["title"],
            "faithfulness_score": content.faithfulness_score if content.faithfulness_score is not None else 0,
            "fluency_score": content.fluency_score if content.fluency_score is not None else 0,
            "overall_score": content.overall_score if content.overall_score is not None else 0,
        }
        return result
    except json.JSONDecodeError:
        print(f"[WARN] JSON parse error for poem {poem['id']}", file=sys.stderr)
        return {
            "id": poem["id"],
            "title": poem["title"],
            "faithfulness_score": None,
            "fluency_score": None,
            "overall_score": None
        }

# ─── Build poem objects (pair verses by index) ─────────────────────────────
def collect_poems(gt, pred_lookup):
    poems = []
    for s in gt:
        pid, title = s["id"], s.get("title","")
        gt_v  = s.get("explanation", [])
        pr_v  = pred_lookup.get(pid, [])
        if not gt_v or not pr_v: continue
        n = min(len(gt_v), len(pr_v))
        poems.append({
            "id": pid,
            "title": title,
            "ground_truth": [
                {"v": i+1, "text": norm(gt_v[i]["explanation"])} for i in range(n)
            ],
            "generated": [
                {"v": i+1, "text": norm(pr_v[i]["explanation"])} for i in range(n)
            ]
        })
    return poems

# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    gt   = load(GT_FILE)
    pred = load(PRED_FILE)
    poems = collect_poems(gt, build_pred_lookup(pred))

    # resume: load existing + shrink list
    results = load(OUT_FILE) if Path(OUT_FILE).exists() else []
    done_ids = {r["id"] for r in results}
    todo = [p for p in poems if p["id"] not in done_ids]

    logging.info(f"{len(todo)} poems to score (already done: {len(done_ids)})")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_poem, p): p["id"] for p in todo}
        for fut in as_completed(futures):
            pid = futures[fut]
            res = fut.result()
            if res:
                with lock:
                    results.append(res)
                    save_atomic(results, OUT_FILE)
                logging.info(f"[{pid}] saved.")
            time.sleep(REQUEST_PAUSE)  # mild pacing for RPM

    logging.info("All finished.")

if __name__ == "__main__":
    main()