from transformers import AutoModelForSequenceClassification


def build_model(model_name: str, num_labels: int = 2):
    """Build a BERT sequence classifier for binary sentiment analysis."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
