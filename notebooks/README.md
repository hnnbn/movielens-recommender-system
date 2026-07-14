# Notebooks

## `recommendation_system_baseline.ipynb`

Baseline and comparison notebook covering:

- MovieLens data loading
- Item-based collaborative filtering
- Mean-centered item CF
- Top-K neighbor tuning
- SVD, SVD++, NMF experiments
- Weighted hybrid and bagging comparison
- Test evaluation with RMSE, MAE, Precision@10, and Recall@10

## `ensemble_training.ipynb`

Advanced ensemble notebook covering:

- Manual SVD and SVD++ implementation
- Integrated hybrid model
- Weighted ensemble search
- Bagging and boosting-style sequential ensemble
- Model persistence
- Long-tail weighted ensemble
- Catalog coverage and diversity analysis

## Local Path Note

The notebooks were originally written in Google Colab. If running locally, update Colab Drive paths such as:

```python
"/content/drive/MyDrive/movie/ua.base"
```

to:

```python
"data/raw/ml-100k/ua.base"
```
