# distilbert-imdb-sentiment
# Fine-Tuning DistilBERT for Sentiment Analysis

Fine-tuned `distilbert-base-uncased` on the IMDb movie review dataset for
binary sentiment classification (positive/negative) using Hugging Face
Transformers and PyTorch.

## Overview
This project applies transfer learning: a pre-trained language model is
adapted to a specific task using a relatively small amount of labeled data.

## Workflow
1. **Setup:** Hugging Face `transformers`, `datasets`, `evaluate`, and
   PyTorch, trained on a Google Colab NVIDIA T4 GPU.
2. **Data:** IMDb dataset with 25,000 training and 25,000 test reviews.
   Analyzed token lengths to choose a truncation limit.
3. **Tokenization:** `AutoTokenizer` for `distilbert-base-uncased`, with
   padding and truncation to a maximum of 256 tokens.
4. **Model:** `AutoModelForSequenceClassification` (pre-trained DistilBERT
   plus a classification head).
5. **Training:** Validated the setup on a small subset, then trained for
   2 epochs with the `Trainer` API (learning rate 2e-5, weight decay 0.01,
   fp16 mixed precision).
6. **Evaluation and inference:** Evaluated on the full test set and built a
   function that predicts sentiment for new sentences.

## Results (25,000 test reviews)

| Metric | Score |
|:---|:---:|
| Accuracy | 91.32% |
| F1-Score | 91.41% |

| Class | Precision | Recall |
|:---|:---:|:---:|
| Negative | 92% | 90% |
| Positive | 90% | 92% |

## Example Predictions

| Sentence | Predicted | Correct? |
|:---|:---:|:---:|
| (paste a clearly positive example) | Positive | Yes |
| (paste a clearly negative example) | Negative | Yes |
| (paste your sarcasm example) | ... | ... |
| (paste your mixed-sentiment example) | ... | ... |

## How Transfer Learning Is Applied
DistilBERT was pre-trained on large general text corpora, so it already
captures grammar, word meaning, and context. Fine-tuning on IMDb only
needs to teach it the specific task of judging sentiment, which takes far
less data and compute than training a language model from scratch.

## Limitations
- Reviews are truncated to 256 tokens, so information in long reviews
  can be lost.
- Trained only on movie reviews, so it may not generalize to other domains
  such as product reviews or tweets.
- Struggles with sarcasm and mixed sentiment [edit to match what you saw].
- Only 2 epochs and no hyperparameter search.

## Future Work
- Error analysis of misclassified reviews
- Try a longer max length, or compare with BERT/RoBERTa
- Test on a different domain to check generalization

## Tech Stack
Python, Hugging Face Transformers and Datasets, PyTorch, scikit-learn,
Evaluate

## How to Run
Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshara-0812/distilbert-imdb-sentiment/blob/main/YOUR_NOTEBOOK_NAME.ipynb)

Or locally (a GPU is recommended):

    git clone https://github.com/Akshara-0812/distilbert-imdb-sentiment.git
    cd distilbert-imdb-sentiment
    pip install -r requirements.txt
    jupyter notebook
