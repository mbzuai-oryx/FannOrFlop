import json
import statistics # Import the statistics module for stdev

def calculate_scores_with_sd(model_data):
    """
    Calculates the average and standard deviation for faithfulness and fluency
    scores from a list of poem evaluations.

    Args:
        model_data (list): A list of dictionaries, where each dictionary contains
                           evaluation scores for a poem, including 'faithfulness_score'
                           and 'fluency_score'.

    Returns:
        dict: A dictionary containing the average and standard deviation for
              'faithfulness_score' and 'fluency_score'. Returns None for a
              metric if it cannot be calculated (e.g., insufficient data points
              for SD, or missing scores).
    """
    if not model_data:
        print("Input data is empty. Cannot calculate scores.")
        return None

    faithfulness_scores = []
    fluency_scores = []

    for entry in model_data:
        faithfulness = entry.get('faithfulness_score')
        fluency = entry.get('fluency_score')

        # Validate and collect faithfulness scores
        if faithfulness is not None:
            if isinstance(faithfulness, (int, float)):
                faithfulness_scores.append(faithfulness)
            else:
                print(f"Warning: Non-numeric faithfulness score found in entry: {entry.get('id', 'Unknown ID')}. Skipping this score.")
        else:
            print(f"Warning: Missing faithfulness score in entry: {entry.get('id', 'Unknown ID')}. Skipping this score.")

        # Validate and collect fluency scores
        if fluency is not None:
            if isinstance(fluency, (int, float)):
                fluency_scores.append(fluency)
            else:
                print(f"Warning: Non-numeric fluency score found in entry: {entry.get('id', 'Unknown ID')}. Skipping this score.")
        else:
            print(f"Warning: Missing fluency score in entry: {entry.get('id', 'Unknown ID')}. Skipping this score.")

    results = {}

    # Calculate for faithfulness
    if faithfulness_scores:
        results['average_faithfulness'] = statistics.mean(faithfulness_scores)
        if len(faithfulness_scores) >= 2: # Standard deviation requires at least 2 data points
            results['sd_faithfulness'] = statistics.stdev(faithfulness_scores)
        else:
            results['sd_faithfulness'] = None # Not enough data for SD
            print(f"Warning: Not enough data points ({len(faithfulness_scores)}) to calculate standard deviation for faithfulness.")
    else:
        results['average_faithfulness'] = None
        results['sd_faithfulness'] = None
        print("No valid faithfulness scores found to calculate average or SD.")

    # Calculate for fluency
    if fluency_scores:
        results['average_fluency'] = statistics.mean(fluency_scores)
        if len(fluency_scores) >= 2: # Standard deviation requires at least 2 data points
            results['sd_fluency'] = statistics.stdev(fluency_scores)
        else:
            results['sd_fluency'] = None # Not enough data for SD
            print(f"Warning: Not enough data points ({len(fluency_scores)}) to calculate standard deviation for fluency.")
    else:
        results['average_fluency'] = None
        results['sd_fluency'] = None
        print("No valid fluency scores found to calculate average or SD.")

    return results

# --- Example Usage (similar to your script structure) ---
if __name__ == "__main__":


    model_jsons = {
        "gpt-4o": "gpt-4o-results.json",
        "gpt-4o-mini": "gpt-4o-mini-results.json",
        "gemini_1_5_pro": "gemini_1_5_pro-results.json",
        "gemini_2_0_flash": "gemini_2_0_flash-results.json",
        "gemini_2_5_flash": "gemini_2_5_flash-results.json",
        "deepseek_R1": "deepseek_R1-results.json",
        "deepseek_V3": "deepseek_v3-results.json",
        "llama_3_3_70B": "llama_3_3_70b-results.json",
        "aya_expanse": "aya_expanse_32b-results.json"
    }
    
    for model_name, json_file in model_jsons.items():
        try:
            with open(json_file, "r") as flp:
                model_evaluations_data = json.load(flp)
        except FileNotFoundError:
            print(f"Error: File not found {json_file} for model {model_name}")
            model_evaluations_data = []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {model_name} from {json_file}: {e}")
            model_evaluations_data = []
    
        if model_evaluations_data:
            scores = calculate_scores_with_sd(model_evaluations_data)
            if scores:
                print(f"\n--- Scores for {model_name} ---")
    
                avg_faith = scores.get('average_faithfulness')
                sd_faith = scores.get('sd_faithfulness')
                if avg_faith is not None:
                    sd_faith_str = f"{sd_faith:.2f}" if sd_faith is not None else "N/A"
                    print(f"Average Faithfulness: {avg_faith:.2f} (SD: {sd_faith_str})")
                else:
                    print("Average Faithfulness: Not calculable")
    
                avg_flu = scores.get('average_fluency')
                sd_flu = scores.get('sd_fluency')
                if avg_flu is not None:
                    sd_flu_str = f"{sd_flu:.2f}" if sd_flu is not None else "N/A"
                    print(f"Average Fluency: {avg_flu:.2f} (SD: {sd_flu_str})")
                else:
                    print("Average Fluency: Not calculable")
    
                print("="*30)


