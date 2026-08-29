from sklearn.model_selection import train_test_split
from src.ingestion.load_financial import load_financial_data

data = load_financial_data()
texts = [row["text"] for row in data]
labels = [row["label"] for row in data]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train size: {len(train_texts)}")
print(f"Test size: {len(test_texts)}")