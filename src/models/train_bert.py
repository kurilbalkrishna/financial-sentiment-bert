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

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

print("Tokenization complete.")
print(f"Sample input_ids length: {len(train_encodings['input_ids'][0])}")