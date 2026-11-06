# Movie Review Sentiment Classification with BERT

[Portfolio Page](https://app.notion.com/p/3e968564df5a81a5ab3ff6a24f630d65)

Fine-tuned a pre-trained BERT model for binary sentiment classification on movie reviews.

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

Both learning rate and epoch count changed between settings; the training-time difference also cannot be explained without the original runtime environment records.

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

## Data and Third-party Resources

- **Dataset:** the MR movie-review dataset files are supplied separately and are subject to the original dataset owner's usage terms.
- **Model:** `bert-base-uncased` is downloaded from its publisher; consult the model card and licensing information.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
