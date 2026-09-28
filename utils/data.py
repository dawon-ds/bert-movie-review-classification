from datasets import load_dataset


def load_mr_splits(train_file: str, val_file: str, test_file: str):
    return load_dataset(
        "csv",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        },
    )
