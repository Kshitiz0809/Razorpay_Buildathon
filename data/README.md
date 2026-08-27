# Data provenance

**Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, `mlg-ulb/creditcardfraud`).

Originally released by the Machine Learning Group at ULB (Université Libre
de Bruxelles) in collaboration with Worldline, from a research
collaboration on big data mining and fraud detection.

- 284,807 transactions by European cardholders over two days in September 2013.
- 492 labeled frauds (0.172% positive rate).
- Features `V1`..`V28` are the output of a PCA transformation applied by the
  original authors to protect cardholder confidentiality — the underlying
  raw features and their real-world meaning are not available, and no
  attempt is made in this project to reconstruct or reverse them.
- `Time`: seconds elapsed between each transaction and the first transaction
  in the dataset. `Amount`: transaction amount, currency unspecified by the
  dataset authors (treated as USD in `configs/cost_config.yaml` purely as a
  documented convention).
- `Class`: 1 = fraud, 0 = legitimate.

## Download

Requires a Kaggle account API token (`kaggle.json`) placed at
`~/.kaggle/kaggle.json`. Then, from the project root:

```
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

This produces `data/raw/creditcard.csv` (~150MB), which is gitignored and
not committed to the repository.

## A note on generalization

This dataset is European cardholder data from 2013. Fraud patterns and
transaction distributions in an Indian payments/BFSI context today will
differ. This project treats the dataset as a rigorous **methodology
demonstration** — split discipline, calibration, cost-sensitive
thresholding, explainability — not as a claim that the trained model itself
is ready to deploy against Indian merchant traffic without retraining on
representative data. See `MODEL_CARD.md`.
