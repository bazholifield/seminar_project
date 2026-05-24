# Climate Stance Detection

A small-scale replication of [Detecting Stance in Media on Global Warming (Luo et al., 2020)](https://aclanthology.org/2020.findings-emnlp.296/), transposing their methodology from news media articles to Twitter. The original study used a BERT classifier to analyze how linguistic devices are deployed differently in pro- and anti-global warming media, finding that GW-skeptical sources use disproportionately more opponent-doubting language than self-affirming language. This project tests whether the same pattern holds for social media.

Built as a junior-year seminar project at Northwestern University (Linguistics / Computational Linguistics).

---

## Overview

Luo et al. identified a key asymmetry in climate discourse: both sides use similar categories of opinion-framing devices (self-affirmation and opponent-doubting), but GW-skeptical media relies more heavily on the latter. Their framework was developed on long-form media articles using BERT.

This replication asks: does the same asymmetry appear in short-form social media text? And can a lighter model (DistilBERT) detect stance from the extracted linguistic devices alone? If it fails, does it fail in a linguistically meaningful way?

To answer it, the pipeline:
1. Extracts and POS-tags content words (verbs, adjectives, nouns) from each tweet
2. Fine-tunes a DistilBERT classifier to predict stance from tweet text
3. Runs the classifier on the extracted keyword sets and compares predicted stance against ground-truth labels
4. Applies custom affirming and doubt-signaling lexicons to measure opinion-framing device usage across stances

---

## Key Findings

Results are consistent with Luo et al.: the same asymmetry between self-affirming and opponent-doubting language observed in news media holds for Twitter. GW-skeptic tweets use disproportionately more opponent-doubting linguistic devices, while GW-affirming tweets lean more toward self-affirming language. The misclassification pattern additionally supports this: tweets falsely classified as GW-skeptic tend to use more opponent-doubting devices, and those falsely classified as GW-affirming tend to use more self-affirming ones, suggesting the model is picking up on the right features even when it gets the label wrong.

**Dataset:** 43,943 tweets across four stance labels — Pro (52.3%), News (21.1%), Neutral (17.6%), Anti (9.1%)

**Adjective density (evaluative language proxy):**

| Stance | Adj/keyword ratio |
|--------|------------------|
| Anti   | 0.149            |
| Neutral | 0.147           |
| Pro    | 0.118            |
| News   | 0.113            |

Anti-stance tweets use adjectives at a ~26% higher rate than Pro-stance tweets, consistent with heavier reliance on evaluative and inflammatory framing.

**Distinctive vocabulary:**

Anti-stance tweets are disproportionately associated with words like *hoax*, *scam*, *fraud*, *manipulated*, *alarmists*, *liberal*, *liberals*, *cooling*, and *freezing* - a pattern suggesting both conspiratorial framing and motivated skepticism about the scientific consensus.

Pro-stance tweets are disproportionately associated with words like *impacts*, *effects*, *health*, *action*, *protect*, *future*, *denial*, and *tackle* -language oriented around consequences and urgency.

**POS patterns:**

Anti-stance tweets average more adjectives per tweet (2.1) than any other group. Pro-stance tweets show higher verb usage on average (3.2 vs 3.0 for Anti), consistent with action- and event-oriented framing.

---

## Pipeline

### `pos_analysis.py`
Tokenizes each tweet using NLTK with a custom MWE (multi-word expression) tokenizer for domain-relevant phrases (e.g., *climate_scientist*, *peer_review*, *nobel_laureate*). Extracts verbs, adjectives, and nouns via POS tagging and writes results to `analysis.csv`.

### `pos_finetune.py`
Fine-tunes `DistilBertForSequenceClassification` on the full tweet corpus for 4-class stance classification (Pro, Anti, Neutral, News). Uses AdamW optimizer with a learning rate of 5e-5 and a step LR scheduler over 3 epochs.

### `pos_sentiment_vocab.py`
Loads the fine-tuned model and runs inference on the extracted keyword sets from `analysis.csv`. Compares predicted vs. ground-truth stance labels. Applies custom affirm/doubt lexicons to compute per-stance opinion-framing ratios.

---

## Lexicons

Five custom word lists in the repo root drive the opinion-framing analysis:

- `AFFIRM_WORDS.txt` — affirming language (e.g., *confirmed*, *proven*, *scientific*)
- `DOUBT_WORDS.txt` — doubt-signaling language (e.g., *hoax*, *myth*, *alleged*)
- `PRO_WORDS.txt` — climate-affirming vocabulary
- `ANTI_WORDS.txt` — climate-denying vocabulary
- `SCI_WORDS.txt` — scientific/academic register vocabulary
- `VERBS_WORDS.txt` — action verbs relevant to climate discourse

---

## Data

**Dataset:** [Twitter Climate Change Sentiment Dataset](https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset) (Kaggle, edqian)

**Labels:**
- `2` — News (factual reporting)
- `1` — Pro (affirms climate change)
- `0` — Neutral
- `-1` — Anti (denies or doubts climate change)

`analysis.csv` is the output of `pos_analysis.py` and contains one row per tweet with columns: `id`, `sentiment`, `verbs`, `adjectives`, `nouns`, `keywords`.

---

## Setup

```bash
# Python 3.10 recommended
pip install pandas nltk transformers torch numpy

python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

**To run the full pipeline:**

```bash
# Step 1: POS extraction
python pos_analysis.py

# Step 2: Fine-tune DistilBERT (requires GPU recommended)
python pos_finetune.py

# Step 3: Lexicon analysis and stance prediction
python pos_sentiment_vocab.py
```

Note: `pos_finetune.py` saves the trained model to `fine_tuned_distilbert_sentiment/`. This checkpoint is not included in the repo due to file size; run Step 2 to generate it before running Step 3.

---

## Reference

Luo, Y., Hardmeier, C., & Riedl, J. (2020). Detecting Stance in Media on Global Warming. *Findings of EMNLP 2020*. https://aclanthology.org/2020.findings-emnlp.296/

---

## Limitations

- The Anti-stance class is substantially underrepresented (9.1% of the dataset), which likely affects classifier performance on that class
- The dataset originates from a specific time window (2016–around the US election), which may limit generalizability
- The MWE tokenizer covers a limited set of domain phrases; a more comprehensive treatment would improve keyword quality
- Lexicons were constructed manually and are not exhaustive
