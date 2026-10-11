# Movie Review Sentiment Classification with BERT

Fine-tuned a pre-trained BERT model for binary sentiment classification on movie reviews.

[Portfolio](https://incredible-march-0ef.notion.site/Movie-Review-Sentiment-Classification-with-BERT-3e968564df5a81a5ab3ff6a24f630d65)

## Project Overview

- **Task:** Movie review sentiment classification
- **Dataset:** MR (Movie Review)
- **Model:** `bert-base-uncased`
- **Classes:** Positive / Negative
- **Framework:** PyTorch + Hugging Face Transformers
- **Batch size:** 8
- **Max sequence length:** 128
- **Warmup ratio:** 0.1

## Approach

The CSV loader creates separate training, validation, and test datasets. Reviews are tokenized with truncation at 128 tokens and dynamically padded within each batch. Hugging Face Trainer fine-tunes the two-class model and evaluates validation accuracy each epoch.

The default configuration corresponds to Setting #2. To run Setting #1, change learning rate to `5e-5` and epochs to `3`, and use a separate output directory.

Prepare CSV files at `data/mr_train.csv`, `data/mr_dev.csv`, and `data/mr_test.csv`. The CSV files must include `sentence` and `label` columns, with integer labels 0 or 1. Keep the same positive/negative label mapping across all three splits.

Model and tokenizer files are saved under `runs/bert_mr/final/`. Use this directory as the checkpoint path for the evaluation and prediction commands.

## Results

| Setting | Learning Rate | Epochs | Accuracy | Training Time |
|---|---:|---:|---:|---:|
| #1 | 5e-5 | 3 | 86.98% | 6.10 min |
| #2 | 3e-5 | 4 | 87.50% | 290.8 min |

### Interpretation

Setting #2 improves reported accuracy by **0.52 percentage points**. Both learning rate and epoch count change, so the comparison cannot isolate either factor. The large training-time difference cannot be attributed to those settings alone without hardware and runtime records.

## Repository Structure

```text
bert-movie-review-classification/
├── config/
│   └── bert_mr.yaml
├── models/
│   ├── __init__.py
│   └── bert_classifier.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── utils/
│   ├── __init__.py
│   └── data.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python -m scripts.train --config config/bert_mr.yaml
python -m scripts.evaluate --config config/bert_mr.yaml --checkpoint <checkpoint_path>
python -m scripts.predict --checkpoint <checkpoint_path> --text "This movie was great."
```

## Limitations

The current training script saves the final model rather than automatically restoring the highest-validation-accuracy checkpoint.

## Review

The project provided experience with pretrained text-classification models, tokenization, dynamic padding, and configuration-based experiments. Future work could add repeated seeds, per-class metrics, error analysis, and documented runtime conditions.
