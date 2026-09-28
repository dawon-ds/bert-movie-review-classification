import argparse
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    classifier = pipeline("text-classification", model=args.checkpoint, tokenizer=args.checkpoint)
    print(classifier(args.text))


if __name__ == "__main__":
    main()
