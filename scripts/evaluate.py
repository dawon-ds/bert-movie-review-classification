import argparse

import numpy as np
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

from utils import load_mr_splits


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bert_mr.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    dataset = load_mr_splits(
        cfg["data"]["train_file"], cfg["data"]["val_file"], cfg["data"]["test_file"]
    )

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=cfg["data"]["max_seq_length"])

    test_set = dataset["test"].map(tokenize, batched=True)
    trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    output = trainer.predict(test_set)
    predictions = np.argmax(output.predictions, axis=-1)
    accuracy = float((predictions == np.asarray(test_set["label"])).mean())
    print(f"test_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
