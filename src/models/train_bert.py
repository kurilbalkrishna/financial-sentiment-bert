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

import torch


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Sample item keys: {train_dataset[0].keys()}")