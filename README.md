# Movie Review Sentiment Classification with BERT

Fine-tuned a pre-trained BERT model for binary sentiment classification on movie reviews.

## Overview
- **Task:** Movie review sentiment classification
- **Dataset:** MR (Movie Review)
- **Model:** `bert-base-uncased`
- **Classes:** Positive / Negative
- **Framework:** PyTorch + Hugging Face Transformers
- **Batch size:** 8
- **Max sequence length:** 128
- **Warmup ratio:** 0.1

## Experiments
| Setting | Learning Rate | Epochs | Accuracy | Training Time |
|---|---:|---:|---:|---:|
| #1 | 5e-5 | 3 | 86.98% | 6.10 min |
| #2 | 3e-5 | 4 | 87.50% | 290.8 min |

## Project Structure
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

## Notes
The original coursework code was reorganized into a reusable portfolio structure. Local absolute paths and hard-coded timestamps were removed.
