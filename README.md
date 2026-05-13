# SME Survey LLM — Validating LLMs as Survey Coders

A research project investigating whether large language models can replace human coders in large-scale qualitative survey analysis, applied to SME (small and medium enterprise) AI adoption data.

Built as part of a PhD application in computational social science / AI research.

---

## Research Question

> Can an LLM achieve inter-rater reliability comparable to human coders when classifying open-ended survey responses?

The key benchmark: if LLM–human agreement (kappa) meets or exceeds human–human agreement (kappa), the research claim holds — LLMs are a credible substitute for human coders at scale.

---

## Dataset

- **500 synthetic SME survey responses** generated across 28 label combinations and 224 unique response templates
- Three classification dimensions:
  - `sentiment` — positive / neutral / negative
  - `adoption_stage` — exploring / piloting / scaling / not_interested
  - `main_barrier` — cost / skills / trust / none

---

## Pipeline

```
generate_dataset.py        # Generate 500 synthetic responses with ground-truth labels
       ↓
llm_classifier.py          # Classify all responses using Claude Haiku (async, prompt caching)
       ↓
evaluate.py                # Compute accuracy, Cohen's kappa, confusion matrices, error analysis
       ↓
prepare_coding_task.py     # Sample 150 responses → generate human coder interface (coder.html)
       ↓
human_coding_analysis.py   # Compare human–human kappa vs LLM–human kappa
```

---

## LLM Results (vs. Ground Truth, n=464)

| Dimension       | Accuracy | Cohen's Kappa | Interpretation    |
|-----------------|----------|---------------|-------------------|
| Sentiment       | 90.3%    | 0.854         | Almost perfect    |
| Adoption stage  | 89.5%    | 0.858         | Almost perfect    |
| Main barrier    | 91.0%    | 0.877         | Almost perfect    |

The classifier uses **Claude Haiku** with prompt caching, JSON prefill, and a concurrency of 10 async workers.

---

## Human Coding Validation (In Progress)

A 150-response subset has been prepared for human coders via a self-contained web interface:

**[Live coding task](https://younesboubbou.github.io/sme-survey-llm/coding_task/coder.html)**

Human coders complete the task in the browser and download a CSV. Once ~5 coder CSVs are collected:

```bash
# Save each file as:
coding_task/coder_<name>.csv

# Then run:
python human_coding_analysis.py
```

This produces:
- Human–human kappa (inter-rater reliability baseline)
- LLM–human kappa (how well the model matches real coders)
- A comparison plot across all three dimensions

---

## Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
# .env
ANTHROPIC_API_KEY=your_key_here
```

Run the full pipeline:

```bash
python generate_dataset.py
python llm_classifier.py
python evaluate.py
python prepare_coding_task.py
```

---

## Project Structure

```
sme-survey-llm/
├── data/
│   ├── sme_survey_responses.csv      # Synthetic dataset with ground-truth labels
│   ├── llm_predictions.csv           # Model classifications
│   └── llm_predictions_fixed.csv     # Cleaned predictions
├── outputs/
│   ├── summary_metrics.png           # Accuracy + kappa bar charts
│   ├── confusion_matrices.png        # Normalised confusion matrices
│   ├── error_analysis.png            # Error breakdown by category
│   └── misclassifications.csv        # Per-response error log
├── coding_task/
│   ├── coder.html                    # Human coder web interface
│   ├── coding_truth.csv              # Ground-truth for the 150 sampled responses
│   └── coder_<name>.csv             # (Expected) human coder submissions
├── generate_dataset.py
├── llm_classifier.py
├── evaluate.py
├── prepare_coding_task.py
├── human_coding_analysis.py
└── requirements.txt
```

---

## Tech Stack

- [Anthropic Python SDK](https://github.com/anthropic/anthropic-sdk-python) — Claude Haiku classifier
- `scikit-learn` — Cohen's kappa, confusion matrices
- `matplotlib` / `numpy` — visualisation
- GitHub Pages — human coder interface hosting
