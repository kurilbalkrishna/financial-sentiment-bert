# Financial Sentiment BERT

Fine-tuning BERT for 3-class sentiment classification (negative / neutral / positive) on financial text, with a follow-up domain-transfer diagnostic to test whether the model generalizes beyond finance.

## Results

| Task | Metric | Score |
|---|---|---|
| Financial sentiment (in-domain) | Accuracy | 82.47% |
| Financial sentiment (in-domain) | F1 | 0.824 |
| IMDB transfer (out-of-domain) | Neutral-class prediction rate | 97.6% |

## Key finding

The model performs well in-domain but **fails to transfer** to a different text domain (IMDB movie reviews): it collapses to predicting the neutral class on 97.6% of examples instead of discriminating positive/negative sentiment. This points to the model learning financial-domain-specific decision boundaries rather than general sentiment features — a meaningful negative result, documented rather than hidden, since it says something real about the limits of the fine-tuned model.

See `src/models/check_predictions.py` for the diagnostic script and `SECTION 9` of `train_bert.py` for the transfer-learning evaluation.

## Project structure

```
.
├── src/
│   ├── ingestion/          # data loading / preprocessing
│   └── models/
│       ├── train_bert.py       # fine-tuning + IMDB transfer test
│       └── check_predictions.py # diagnostic: per-class prediction breakdown
├── results/                 # training checkpoints (gitignored, generated locally)
├── .gitignore
└── README.md
```

## Setup

```bash
pip install transformers datasets torch scikit-learn
```

## Usage

Fine-tune on the financial sentiment dataset and run the IMDB transfer check in one pass:

```bash
python src/models/train_bert.py
```

Inspect prediction distribution / diagnose failure modes on a trained checkpoint:

```bash
python src/models/check_predictions.py
```

## Training setup

- Base model: `bert-base-uncased`
- Epochs: 2
- Batch size: 16 (train and eval)
- Eval / save strategy: per epoch
- Label scheme: `0 = negative`, `1 = neutral`, `2 = positive`

> Dataset: 3-class financial sentiment labels (negative/neutral/positive) — swap in the exact source/citation here if it's not Financial PhraseBank.

## Why this result matters

Reporting an 82% in-domain accuracy without checking transfer would overstate what the model actually learned. The IMDB test exists specifically to check that, and the honest answer — it doesn't generalize — is more useful than a cherry-picked in-domain number for anyone evaluating this as applied ML work.

## Next steps

- [ ] Quantify class-wise precision/recall on IMDB to isolate whether neutral-collapse is a calibration issue or a genuine feature-transfer failure
- [ ] Compare against a domain-adapted baseline (e.g. FinBERT) on the same transfer test
- [ ] Try partial fine-tuning / gradual unfreezing to reduce overfitting to financial vocabulary

## Author

Balkrishna Kuril
[GitHub](https://github.com/kurilbalkrishna) · [LinkedIn](https://linkedin.com/in/balkrishna-kuril-18497531a)