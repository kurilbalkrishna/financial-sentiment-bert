from datasets import load_dataset

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_financial_data():
    raw = load_dataset("descartes100/enhanced-financial-phrasebank")
    flattened = [
        {"text": row["train"]["sentence"], "label": row["train"]["label"]}
        for row in raw["train"]
    ]
    return flattened


if __name__ == "__main__":
    data = load_financial_data()
    print(f"Total rows: {len(data)}")
    print(data[0])
    print("Label meaning:", LABEL_MAP[data[0]["label"]])