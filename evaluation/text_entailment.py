"""
Batch Textual Entailment for Comparing Ground Truth and Model-Generated Poem Explanations
----------------------------------------------------------------------------------------
Processes multiple models' prediction files, checks how closely their generated
explanations align with ground truth explanations using bidirectional textual entailment,
and saves results in model-specific folders.
"""

import re, json, unicodedata, argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── CAMeL Tools Tokenizer ───────────────────────────────────────────────────────
try:
    from camel_tools.tokenizers.word import simple_word_segment as _segment
except ImportError:
    try:
        from camel_tools.tokenizers.word import simple_word_tokenize as _segment
        print("[INFO] camel_tools.tokenizers.word.simple_word_segment not found – using simple_word_tokenize instead.")
    except ImportError:
        _segment = lambda text: text.split() # Basic fallback
        print("[WARN] CAMeL Tools not found. Using basic whitespace tokenization for normalization. This may affect results for Arabic text.")

# ── Text Normalization ───────────────────────────────────────────────────────────
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def norm(text: str) -> str:
    """Normalizes a string by removing diacritics, normalizing NFKC, and stripping whitespace."""
    if not isinstance(text, str):
        text = str(text) # Ensure text is a string
    text = _DIACRITICS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ── Load JSONs ───────────────────────────────────────────────────────────────────
def load_explained_poems(path: Path, explanations_list_key: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Loads poems from a JSON file.
    Assumes the JSON is a list of poem objects. Each poem object has an 'id'
    and a key (explanations_list_key) which contains a list of items (e.g., verses),
    where each item has an 'explanation' string and a 'verse' string.
    """
    poems_data = {}
    try:
        with path.open(encoding="utf-8") as fp:
            data = json.load(fp)
            if not isinstance(data, list):
                print(f"[ERROR] Expected a list of poem objects in {path}, but got {type(data)}")
                return poems_data
                
            for poem_obj in data:
                if not isinstance(poem_obj, dict):
                    print(f"[WARN] Skipping non-dictionary item in {path}: {poem_obj}")
                    continue
                pid = poem_obj.get("id")
                if pid and explanations_list_key in poem_obj:
                    explanations_list = poem_obj[explanations_list_key]
                    if isinstance(explanations_list, list):
                        valid_explanations = []
                        for item_idx, item in enumerate(explanations_list):
                            if isinstance(item, dict) and "explanation" in item:
                                valid_explanations.append({
                                    "verse": str(item.get("verse", f"Verse {item_idx + 1}")), # Ensure verse is string, provide default
                                    "explanation": str(item["explanation"]) # Ensure explanation is string
                                })
                            else:
                                print(f"[WARN] Skipping malformed item in poem {pid}, key '{explanations_list_key}' in {path}: {item}")
                        if valid_explanations:
                             poems_data[pid] = valid_explanations
                    else:
                        print(f"[WARN] Expected a list for key '{explanations_list_key}' in poem {pid} in {path}, but got {type(explanations_list)}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from file: {path}")
    return poems_data

# ── Entailment Model ─────────────────────────────────────────────────────────────
class EntailmentModel:
    def __init__(self, model_name="joeddav/xlm-roberta-large-xnli"):
        """Initializes the tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] EntailmentModel using device: {self.device} for model {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set model to evaluation mode

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Predicts entailment scores between a premise and a hypothesis."""
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=1)[0]
        
        # Determine label indices safely
        # Default to 0, 1, 2 if specific labels aren't in config, which is common for XNLI
        ent_id = self.model.config.label2id.get('entailment', 0)
        neu_id = self.model.config.label2id.get('neutral', 1)
        con_id = self.model.config.label2id.get('contradiction', 2)

        # Verify that indices are unique and within range
        label_ids = {ent_id, neu_id, con_id}
        if len(label_ids) != 3 or not all(0 <= i < probs.size(0) for i in label_ids):
            # Fallback if label mapping is unusual or problematic
            print(f"[WARN] Problem with label2id mapping ({self.model.config.label2id}). Defaulting to 0:ent, 1:neu, 2:con.")
            ent_id, neu_id, con_id = 0, 1, 2
            if probs.size(0) < 3: # Should not happen with standard NLI models
                 print(f"[ERROR] Model output has fewer than 3 probabilities: {probs.size(0)}")
                 return {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0, "error": "Invalid probability output size"}


        return {
            "entailment": probs[ent_id].item(),
            "neutral": probs[neu_id].item(),
            "contradiction": probs[con_id].item()
        }

# ── Evaluation ───────────────────────────────────────────────────────────────────
def evaluate_explanation_closeness(gt_poems_data: Dict[str, List[Dict[str, str]]], 
                                   pred_poems_data: Dict[str, List[Dict[str, str]]],
                                   entailment_model: EntailmentModel) -> (List[Dict[str, Any]], float):
    """
    Compares ground truth explanations with predicted explanations using bidirectional entailment.
    Args:
        gt_poems_data: Dictionary of ground truth poems.
        pred_poems_data: Dictionary of predicted poems.
        entailment_model: Instance of the EntailmentModel class.
    Returns:
        A tuple containing:
            - results: A list of dictionaries with detailed comparison for each poem.
            - overall_avg_bidirectional_score: The average bidirectional entailment score.
    """
    results = []
    all_bidirectional_scores = []

    # Iterate through poem IDs present in both ground truth and predictions
    common_poem_ids = gt_poems_data.keys() & pred_poems_data.keys()
    if not common_poem_ids:
        print("[WARN] No common poem IDs found between ground truth and predictions.")
        return [], 0.0

    for pid in tqdm(common_poem_ids, desc="Comparing Explanations for Poem"):
        gt_explanations = gt_poems_data[pid]
        pred_explanations = pred_poems_data[pid]
        
        poem_result = {"id": pid, "explanation_comparison_results": []}

        num_gt = len(gt_explanations)
        num_pred = len(pred_explanations)

        # Compare explanations up to the minimum number available for the poem
        # This handles cases where a model might not generate explanations for all verses.
        num_to_compare = min(num_gt, num_pred)
        if num_gt != num_pred:
            print(f"[INFO] Poem {pid}: GT explanations: {num_gt}, Pred explanations: {num_pred}. Comparing {num_to_compare} pairs.")
        
        for i in range(num_to_compare):
            gt_item = gt_explanations[i]
            pred_item = pred_explanations[i]

            verse_ref = norm(gt_item.get("verse", pred_item.get("verse", f"Verse {i+1}")))
            
            gt_expl = norm(gt_item["explanation"])
            pred_expl = norm(pred_item["explanation"])

            if not gt_expl or not pred_expl:
                print(f"[WARN] Empty ground truth or predicted explanation for poem {pid}, item {i+1}. Skipping this pair.")
                comparison_entry = {
                    "verse_reference": verse_ref,
                    "ground_truth_explanation": gt_expl,
                    "model_generated_explanation": pred_expl,
                    "error": "Empty explanation text"
                }
                poem_result["explanation_comparison_results"].append(comparison_entry)
                continue

            # 1. Ground Truth Explanation (GTE) entails Predicted Explanation (PE)
            gte_entails_pe_scores = entailment_model.predict(gt_expl, pred_expl)
            gte_entails_pe_label = max(gte_entails_pe_scores, key=lambda k: gte_entails_pe_scores.get(k, -1)) if "error" not in gte_entails_pe_scores else "error"


            # 2. Predicted Explanation (PE) entails Ground Truth Explanation (GTE)
            pe_entails_gte_scores = entailment_model.predict(pred_expl, gt_expl)
            pe_entails_gte_label = max(pe_entails_gte_scores, key=lambda k: pe_entails_gte_scores.get(k, -1)) if "error" not in pe_entails_gte_scores else "error"

            # 3. Bidirectional Entailment Strength
            bidirectional_strength = 0.0
            if "error" not in gte_entails_pe_scores and "error" not in pe_entails_gte_scores:
                bidirectional_strength = (gte_entails_pe_scores.get("entailment", 0.0) + pe_entails_gte_scores.get("entailment", 0.0)) / 2
                all_bidirectional_scores.append(bidirectional_strength)

            comparison_entry = {
                "verse_reference": verse_ref,
                "ground_truth_explanation": gt_expl,
                "model_generated_explanation": pred_expl,
                "gte_entails_pe_scores": gte_entails_pe_scores,
                "gte_entails_pe_label": gte_entails_pe_label,
                "pe_entails_gte_scores": pe_entails_gte_scores,
                "pe_entails_gte_label": pe_entails_gte_label,
                "bidirectional_entailment_strength": bidirectional_strength
            }
            poem_result["explanation_comparison_results"].append(comparison_entry)
        
        if poem_result["explanation_comparison_results"]:
            results.append(poem_result)

    overall_avg_bidirectional_score = sum(all_bidirectional_scores) / len(all_bidirectional_scores) if all_bidirectional_scores else 0.0
    
    return results, overall_avg_bidirectional_score

# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compare ground truth and model-generated poem explanations using bidirectional entailment.")
    parser.add_argument("--gt_file", default="final_poems.json", help="Path to the JSON file containing ground truth poems with explanations.")
    parser.add_argument("--gt_key", default="explanation", help="The key in the GT JSON objects that holds the list of verse/explanation dicts (e.g., 'explanation').")
    parser.add_argument("--pred_key", default="verse_explanations", help="The key in the Pred JSON objects that holds the list of verse/explanation dicts (e.g., 'verse_explanations').")
    parser.add_argument("--base_output_dir", default="./entailments_results", help="Base directory to save model-specific output folders and files.")
    parser.add_argument("--nli_model_name", default="joeddav/xlm-roberta-large-xnli", help="Name of the NLI model from Hugging Face Hub.")

    args = parser.parse_args()

    # --- Define your models and their prediction files here ---
    # The key is the model name (used for folder creation), value is the path to its prediction file.
    models_to_predictions = {
        # "GPT-4o": "gpt_4o_generated_explanation.json",
        # "GPT-4o-mini": "gpt_4o_mini_generated_explanation.json",
        # "Gemini-1_5_pro": "gemini_1_5_pro_generated_explanation.json",
        # "Gemini-2_0_Flash": "gemini_2_0_flash_generated_explanation.json",
        # "Gemini-2_5_Flash": "gemini_2_5_flash_generated_explanation.json",
        # "Llama_3_3_70B": "llama_3_3_70b_generated_explanation.json",
        # "deepseek_v3": "deepseek_v3_generated_explanation.json",
        # "deepseek_r1": "deepseek_v3_generated_explanation.json",
    }
    # --- Make sure to update the paths above to your actual prediction file locations ---


    print(f"[INFO] Loading NLI model: {args.nli_model_name}")
    entailment_model_instance = EntailmentModel(model_name=args.nli_model_name)

    print(f"[INFO] Loading Ground Truth data from: {args.gt_file} (using key: '{args.gt_key}')")
    gt_data = load_explained_poems(Path(args.gt_file), args.gt_key)
    if not gt_data:
        print(f"[ERROR] Ground truth data could not be loaded from {args.gt_file}. Exiting.")
        exit(1)

    base_output_path = Path(args.base_output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True) # Ensure base output directory exists

    overall_summary = {} # To store the main score for each model

    for model_name, pred_file_path_str in models_to_predictions.items():
        print(f"\n--- Processing Model: {model_name} ---")
        pred_file_path = Path(pred_file_path_str)

        if not pred_file_path.exists():
            print(f"[WARN] Prediction file for model '{model_name}' not found at '{pred_file_path_str}'. Skipping.")
            overall_summary[model_name] = {"status": "file_not_found", "score": 0.0}
            continue

        print(f"[INFO] Loading Predictions for '{model_name}' from: {pred_file_path} (using key: '{args.pred_key}')")
        pred_data = load_explained_poems(pred_file_path, args.pred_key)

        if not pred_data:
            print(f"[WARN] Prediction data for model '{model_name}' could not be loaded or is empty. Skipping.")
            overall_summary[model_name] = {"status": "no_data_loaded", "score": 0.0}
            continue
        
        # Create model-specific output directory
        model_output_dir = base_output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = model_output_dir / f"{model_name}_explanation_closeness.json"

        print(f"[INFO] Evaluating closeness for model '{model_name}'...")
        detailed_results, model_overall_score = evaluate_explanation_closeness(
            gt_data,
            pred_data,
            entailment_model_instance
        )
        
        if detailed_results: # Save only if there are results
            with output_json_path.open("w", encoding="utf-8") as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            print(f"[✓] Detailed comparison results for '{model_name}' saved to {output_json_path}")
        else:
            print(f"[INFO] No comparison results generated for model '{model_name}'.")


        print(f"--------------------------------------------------------------------")
        print(f"Model: {model_name} - Overall Average Bidirectional Entailment Strength: {model_overall_score:.4f}")
        print(f"--------------------------------------------------------------------")
        overall_summary[model_name] = {"status": "processed", "score": model_overall_score, "output_file": str(output_json_path)}

    print("\n\n=== Overall Summary of Bidirectional Entailment Scores ===")
    for model_name, summary_data in overall_summary.items():
        if summary_data["status"] == "processed":
            print(f"- {model_name}: {summary_data['score']:.4f} (Results: {summary_data['output_file']})")
        else:
            print(f"- {model_name}: Skipped ({summary_data['status']})")
    print("==========================================================")

