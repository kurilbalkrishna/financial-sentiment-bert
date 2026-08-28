from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sample_text = "The company reported strong quarterly earnings."
tokens = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")

print(tokens)