import argparse
from pathlib import Path

import numpy as np
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from models import build_model
from utils import load_mr_splits


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bert_mr.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    dataset = load_mr_splits(
        cfg["data"]["train_file"],
        cfg["data"]["val_file"],
        cfg["data"]["test_file"],
    )

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg["data"]["max_seq_length"],
        )

    tokenized = dataset.map(tokenize, batched=True)
    model = build_model(cfg["model"]["name"], cfg["model"]["num_labels"])

    output_dir = Path(cfg["output"]["dir"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
