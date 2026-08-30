from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from collections import Counter

# Reload the IMDB data (same subset as before)
imdb = load_dataset("stanfordnlp/imdb")
imdb_test_texts = imdb["test"]["text"][:500]

# Load the saved checkpoint instead of retraining
model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-486")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Run predictions
inputs = tokenizer(imdb_test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

print("Predicted class distribution:", Counter(predictions.tolist()))