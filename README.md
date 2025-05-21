📚 Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding

Fann or Flop is the first comprehensive benchmark designed to evaluate large language models (LLMs) on their ability to understand Arabic poetry. It contains nearly 7,000 poem-explanation pairs covering 12 poetic eras, 21 genres, and multiple meters, providing a culturally rich and linguistically challenging testbed for Arabic NLP.

> 📝 Arxiv Preprint: [Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs (ACL 2025 Submission)](https://arxiv.org/abs/XXXX.XXXXX)  
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
  - **Interpretive Depth**

### 🔹 Human Evaluation
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
```python
coming soon!
```


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


