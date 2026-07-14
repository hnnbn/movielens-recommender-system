# Data

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) from GroupLens.

MovieLens 100K contains 100,000 ratings from about 1,000 users on about 1,700 movies and is commonly used as a benchmark dataset for recommender systems.

## Expected Layout

After downloading and extracting the dataset, keep the files under:

```text
data/raw/ml-100k/
├─ ua.base
├─ ua.test
├─ u.item
└─ ...
```

The notebooks mainly use:

- `ua.base`: training ratings
- `ua.test`: test ratings
- `u.item`: movie metadata and genre indicators

## Why Raw Data Is Not Committed

The raw dataset is intentionally excluded from Git using `.gitignore`.

For portfolio review, this keeps the repository small and makes the data source explicit. To reproduce the project locally, run:

```bash
python scripts/download_movielens_100k.py
```
