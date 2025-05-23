

<div style="margin-top:50px;">
  <h1 style="font-size: 30px; margin: 0;">  📚 Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding
</h1>
</div>


<div  align="center" style="margin-top:10px;"> 
    
  [Wafa Alghallabi](https://huggingface.co/SLMLAH) <sup> * </sup> &nbsp;
  [Ritesh Thawkar](https://in.linkedin.com/in/ritesh-thawkar-b13192233) <sup> * </sup> &nbsp;
  [Sara Ghaboura](https://huggingface.co/SLMLAH) <sup> * </sup> &nbsp;
  [Ketan More](https://scholar.google.com/citations?user=FCgQeoYAAAAJ&hl=en) <sup> * </sup> &nbsp;
  [Omkar Thawakar](https://scholar.google.com/citations?user=flvl5YQAAAAJ&hl=en) <sup> * </sup>  &nbsp;
  <br>
  [Hisham Cholakkal](https://scholar.google.com/citations?hl=en&user=bZ3YBRcAAAAJ) &nbsp;
  [Salman Khan](https://scholar.google.com/citations?hl=en&user=M59O9lkAAAAJ) &nbsp;
  [Rao M. Anwer](https://scholar.google.com/citations?hl=en&user=_KlvMVoAAAAJ)
  <br>
  <br>  
  [![arXiv](https://img.shields.io/badge/arXiv-2502.14865-F6D769)](https://arxiv.org/abs/xxxx.xxxx)
  [![Our Page](https://img.shields.io/badge/Visit-Our%20Page-E7DAB7?style=flat)](https://mbzuai-oryx.github.io/FannOrFlop/)
  [![GitHub issues](https://img.shields.io/github/issues/mbzuai-oryx/Camel-Bench?color=E5D5C1&label=issues&style=flat)](https://github.com/mbzuai-oryx/FannOrFlop/issues)
  [![GitHub stars](https://img.shields.io/github/stars/mbzuai-oryx/TimeTravel?color=FAF1D9&style=flat)](https://github.com/mbzuai-oryx/FannOrFlop/stargazers)
  [![GitHub license](https://img.shields.io/github/license/mbzuai-oryx/Camel-Bench?color=F1E9E3)](https://github.com/mbzuai-oryx/FannOrFlop/blob/main/LICENSE)
  <br>
  <em> <sup> *Equal Contribution  </sup> </em>
  <br>
  <br>
</div>

Fann or Flop is the first comprehensive benchmark designed to evaluate large language models (LLMs) on their ability to understand Arabic poetry. It contains nearly 7,000 poem-explanation pairs covering 12 poetic eras, 21 genres, and multiple meters, providing a culturally rich and linguistically challenging testbed for Arabic NLP.

> 📝 Arxiv Preprint: [Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs](https://arxiv.org/abs/XXXX.XXXXX)  
> 📂 Dataset hosted on: [Hugging Face](https://huggingface.co/datasets/omkarthawakar/FannOrFlop)

---

## ✨ Key Features

- ✅ **Expert-Annotated Explanations:** Verse-level commentary verified by native Arabic scholars.
- 🏛️ **12 Historical Eras:** From Pre-Islamic and Umayyad to Modern poetry.
- 🧠 **Multi-Dimensional Evaluation:** Faithfulness, fluency, metaphor, historical context, and rhetorical awareness.
- 🧾 **Structured Taxonomy:** Each poem tagged with `meter`, `genre`, and `era`.
- 💬 **QA-Style Format:** Ideal for generative and comprehension-based evaluation in LLMs.

---

## 📦 Dataset Structure

Each JSON entry is structured as follows:

| Field             | Type         | Description                                                                 |
|------------------|--------------|-----------------------------------------------------------------------------|
| `id`             | `string`     | Unique poem identifier                                                      |
| `title`          | `string`     | Title of the poem                                                           |
| `author`         | `string`     | Name of the poet                                                            |
| `source`         | `string`     | URL to the poem source                                                      |
| `tags`           | `list[str]`  | List of `meter`, `genre`, and `era`                                         |
| `meter`          | `string`     | Poetic meter (e.g., الكامل, الطويل)                                          |
| `genre`          | `string`     | Genre label (e.g., مدح, رثاء)                                                |
| `era`            | `string`     | Historical literary era (e.g., العصر العباسي)                                |
| `verse_count`    | `int`        | Number of verses                                                            |
| `poem_verses`    | `string`     | Full poem text, numbered and formatted                                      |
| `explanation`    | `list[dict]` | Verse-wise explanation with fields: `verse`, `explanation`                 |
| `raw_explanation`| `string`     | Full explanation in paragraph format                                        |

> Sample entries are available in the [`samples/`](samples/) folder.

---

## 🌍 Taxonomy Overview

The dataset spans 12 major **Arabic poetic eras**:

| Era             | Approx. Time Range     | Example Poets                          |
|------------------|------------------------|-----------------------------------------|
| Pre-Islamic       | ~6th Century           | Imru’ al-Qays, Antarah ibn Shaddad      |
| Umayyad           | 661–750 CE             | Jarir, Al-Farazdaq                      |
| Abbasid           | 750–1258 CE            | Al-Mutanabbi, Abu Nuwas                |
| Andalusian        | 756–1492 CE            | Ibn Zaydun, Ibn Khafaja                |
| Modern            | 19th c. – Present      | Hafiz Ibrahim, Ahmad Shawqi            |
| *(+7 more eras...)* | *See paper for full list* | -                                   |

Each poem is assigned its literary context through expert-verified metadata.

---

## 🧪 Evaluation Protocol

We provide an evaluation framework using:

### 🔹 Automatic Metrics
- **BLEU / chrF++** for lexical overlap
- **BERTScore** (Arabic transformer) for semantic similarity
- **Textual Entailment** using mDeBERTa (NLI)

### 🔹 LLM-as-Judge
- GPT-4o used to evaluate:
  - **Faithfulness / Consistency**
  - **Fluency / Grammaticality**

### 🔹 Human Evaluation
- **Interpretive Depth**
    - Rubric includes:
        - Literal Meaning (0–1)
        - Thematic / Emotional Depth (0–2)
        - Cultural Context (0–2)
        - Literary Devices (0–3)
        - Expressiveness / Coherence (0–2)

---

## 📥 Download

```bash
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("omkarthawakar/FannOrFlop")
```

## 🧪 Evaluation Suit
The `evaluation/` directory contains scripts to reproduce the benchmark results and evaluate your own models. 

### General Setup

1.  **Navigate to the evaluation directory:**
    ```bash
    cd evaluation
    ```
2.  **Dependencies:** Ensure you have Python 3.x installed. Install necessary packages. It's recommended to use a virtual environment.
    ```bash
    pip install torch transformers evaluate scikit-learn numpy openai camel-tools tqdm
    ```
    (Note: `camel-tools` is crucial for Arabic text processing.)

3.  **Ground Truth Data:** The primary ground truth file is `FannOrFlop.json`. Most scripts expect this file to be present in the `evaluation/` directory or for its path to be configured within the script or via command-line arguments.

4.  **Model Prediction Files:** Your model's generated explanations should be in a JSON format. Each file should contain a list of poem objects. Each poem object must include an `"id"` and a key containing a list of verse-explanation pairs (typically `"verse_explanations"`).

    **Sample Model Prediction JSON (`your_model_explanations.json`):**
    ```python
    [
      {
        "id": "poem_5123",
        "title": "خانَ عَهدي مُعاوِداً خَونَ عَهدي", // Optional, but good for reference
        // Other metadata like genre, meter, author can be included
        "verse_explanations": [
          {
            "verse": "خـانَ عَهـدي مُعـاوِداً خَـونَ عَهـدي\nمَــن لَــهُ خُلَّــتي وَخــالِصُ وُدّي",
            "explanation": "Generated explanation for verse 1..."
          },
          {
            "verse": "بـانَ بِالحُسـنِ وَحـدَهُ لَـم يُنـازِع\nهُ شـَريكٌ وَبِنـتُ بِـالبَثِّ وَحـدي",
            "explanation": "Generated explanation for verse 2..."
          }
          // ... more verses for this poem
        ]
      }
      // ... more poems
    ]
    ```

### Running Evaluation Scripts

All commands below assume you are in the `evaluation/` directory.

**1. BERTScore (`bertscore.py`)**

*   **Purpose:** Calculates BERTScore (Precision, Recall, F1) using AraBERT for semantic similarity.
*   **Configuration:** Modify the `modeljsons` dictionary within `bertscore.py` to include your model's name and the path to its prediction JSON file. Ensure `gtjson` points to `FannOrFlop.json`.
    ```python
    # Example in bertscore.py
    modeljsons = {
        "YourModelName": "path/to/your_model_explanations.json",
    }
    gtjson = "FannOrFlop.json" # Or correct path
    ```
*   **Usage:**
    ```bash
    python bertscore.py
    ```
*   **Output:** Prints macro-averaged Precision, Recall, and F1-score to the console.

**2. BLEU (`bleu.py`)**

*   **Purpose:** Calculates BLEU, Coverage, and BLEU*Coverage for lexical overlap.
*   **Configuration:** Inside `bleu.py`, update the `gtjson`, `predjson`, and `modelname` variables in the `if __name__ == "__main__":` block.
    ```python
    # Example in bleu.py
    gtjson = "FannOrFlop.json"
    predjson = "path/to/your_model_explanations.json"
    modelname = "YourModelName"
    ```
*   **Usage:**
    ```bash
    python bleu.py
    ```
*   **Output:** Prints macro-averaged BLEU, Coverage, and BLEU*Coverage to the console.

**3. chrF Score (`chrf_score.py`)**

*   **Purpose:** Calculates chrF, Coverage, and chrF*Coverage (character n-gram metric).
*   **Configuration:** Modify the `modeljsons` dictionary within `chrf_score.py` similarly to `bertscore.py`. Ensure `gtjson` points to `FannOrFlop.json`.
*   **Usage:**
    ```bash
    python chrf_score.py
    ```
*   **Output:** Prints macro-averaged chrF, Coverage, and chrF*Coverage to the console.

**4. LLM-as-Judge Evaluation (`judge_eval.py`)**

*   **Purpose:** Uses an LLM (e.g., GPT-4o) to evaluate Faithfulness, Fluency, and Overall scores.
*   **Prerequisites:** Set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY='your_api_key_here'
    ```
*   **Configuration:** In `judge_eval.py`, modify the following variables at the top of the script:
    *   `MODEL_NAME`: A name for your model.
    *   `PREDICTIONS_FILE`: Path to your model's prediction JSON file.
    *   `GROUND_TRUTH_FILE`: Path to `FannOrFlop.json` (default is `FannOrFlop.json`).
    *   `LLM_JUDGE_MODEL`: The LLM to use for judging (e.g., "gpt-4o", "gpt-3.5-turbo").
*   **Usage:**
    ```bash
    python judge_eval.py
    ```
*   **Output:** Saves detailed scores to a JSON file in the `judge_results/` directory (e.g., `judge_results/YourModelName-results.json`) and prints progress.

**5. Average LLM Judge Scores (`get_average_scores_for_llm_judge.py`)**

*   **Purpose:** Calculates average and standard deviation for scores generated by `judge_eval.py`.
*   **Prerequisites:** Run `judge_eval.py` first to generate result files in `judge_results/`.
*   **Usage:**
    ```bash
    python get_average_scores_for_llm_judge.py
    ```
*   **Output:** Prints average Faithfulness and Fluency scores (with SD) to the console for each model found in `judge_results/`.

**6. Textual Entailment (`text_entailment.py`)**

*   **Purpose:** Calculates bidirectional textual entailment scores between ground truth and generated explanations.
*   **Configuration:** 
    1.  Edit `text_entailment.py` and update the `models_to_predictions` dictionary to map your model names to their prediction JSON file paths.
        ```python
        # Example in text_entailment.py
        models_to_predictions = {
            "YourModelName": "path/to/your_model_explanations.json",
            # Add other models if evaluating multiple
        }
        ```
    2.  The script uses command-line arguments for other configurations. Key arguments:
        *   `--gt_file`: Path to the ground truth JSON (default: `FannOrFlop.json`).
        *   `--gt_key`: Key in ground truth for explanations list (default: `explanation`).
        *   `--pred_key`: Key in prediction files for explanations list (default: `verse_explanations`).
        *   `--base_output_dir`: Directory to save detailed results (default: `explanation_closeness_results`).
*   **Usage (example):**
    ```bash
    python text_entailment.py --gt_file FannOrFlop.json --base_output_dir results/entailment_scores
    ```
*   **Output:** Saves detailed JSON results per model in subdirectories of `base_output_dir`. Prints overall summary scores to the console.


## 📊 Leaderboard (Sample Results)


| **Model**                               | **BLEU**   | **chrF(++)** | **BERTScore** | **Textual Entailment** | **Faithfulness / Consistency** | **Fluency / Grammaticality** | **Interpretive Depth** |
| --------------------------------------- | ---------- | ------------ | ------------- | ---------------------- | ------------------------------ | ---------------------------- | ---------------------- |
| **Closed Models**                       |            |              |               |                        |                                |                              |                        |
| GPT-4o-2024-08-06 (OpenAI, 2024)        | 0.0395     | **0.2882**   | **0.6410**    | 0.6775                 | 3.92 (± 0.99)                  | **4.96 (± 0.20)**            | **7.52**               |
| GPT-4o-mini-2024-07-18 (OpenAI, 2024)   | 0.0395     | 0.2542       | 0.6124        | 0.4383                 | 2.91 (± 0.75)                  | 4.28 (± 0.57)                | 7.50                   |
| Gemini-2.5-Flash (AI, 2025b)            | 0.0153     | 0.2618       | 0.6319        | **0.7475**             | **4.25 (± 1.00)**              | **4.98 (± 0.16)**            | 7.22                   |
| Gemini-2.0-Flash (AI, 2025a)            | 0.0395     | 0.2618       | 0.6393        | 0.7154                 | 3.99 (± 1.04)                  | 4.95 (± 0.22)                | 6.50                   |
| Gemini-1.5-Pro (Reid et al., 2024)      | 0.0395     | 0.2618       | 0.6333        | 0.6180                 | 3.59 (± 1.00)                  | 4.80 (± 0.41)                | 5.38                   |
| Fanar-Star (Team et al., 2025)          | 0.0138     | 0.1538       | 0.5677        | 0.6468                 | 2.16 (± 0.92)                  | 3.40 (± 0.76)                | 2.88                   |
| **Open Models**                         |            |              |               |                        |                                |                              |                        |
| Deepseek-V3 (Liu et al., 2024)          | 0.0395     | 0.2771       | 0.6335        | 0.5117                 | 3.36 (± 0.91)                  | **4.98 (± 0.16)**            | 4.75                   |
| Deepseek-R1 (Guo et al., 2025)          | 0.0395     | 0.2771       | 0.6335        | 0.5117                 | 3.38 (± 0.92)                  | **4.98 (± 0.16)**            | 4.25                   |
| Llama-3.3-70B (Meta AI, 2024)           | 0.0153     | 0.2618       | 0.6393        | 0.5364                 | 2.51 (± 0.90)                  | 3.37 (± 0.73)                | 7.20                   |
| Qwen-3 (Team, 2025)                     | 0.0296     | **0.2837**   | 0.6158        | 0.6468                 | 3.98 (± 0.90)                  | 4.73 (± 0.45)                | 6.50                   |
| Aya-Expanse (Dang et al., 2024)         | 0.0329     | 0.2771       | 0.6328        | 0.6468                 | 3.76 (± 0.90)                  | 4.68 (± 0.47)                | 5.88                   |
| Jais (Sengupta et al., 2023)            | 0.0312     | 0.2698       | 0.6245        | 0.6023                 | 3.21 (± 0.88)                  | 4.35 (± 0.52)                | 5.35                   |
| ALLaM-7B (Bari et al., 2024)            | 0.0119     | 0.0463       | 0.5375        | 0.5997                 | 1.32 (± 0.62)                  | 2.11 (± 0.89)                | 3.12                   |
| AceGPT-v2-70B-Chat (Huang et al., 2023) | **0.0402** | 0.0412       | 0.5759        | 0.6061                 | 2.52 (± 0.91)                  | 3.46 (± 0.95)                | 4.12                   |

---

💬 Citation

Coming soon!


