# MovieLens Recommendation System

Portfolio project for building and evaluating a movie recommender system on MovieLens 100K. The project compares item-based neighborhood collaborative filtering, model-based collaborative filtering, weighted ensembles, and long-tail reranking.

The main goal is not only to reduce rating prediction error, but also to understand the trade-off between accuracy, catalog coverage, and recommendation diversity.

## Project Overview

- **Dataset**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) ratings
- **Task**: user-item rating prediction and top-N recommendation
- **Metrics**: RMSE, MAE, Kendall Tau, Catalog Coverage, Genre Diversity
- **Core approach**: Item-based CF + model-based CF + weighted ensemble
- **Main result**: long-tail reranking improves catalog coverage with a controlled accuracy trade-off

## Repository Structure

```text
movie-recommender-portfolio/
|-- README.md
|-- requirements.txt
|-- data/
|   `-- README.md
|-- docs/
|   |-- presentation.pdf
|   `-- project-summary.md
|-- notebooks/
|   |-- README.md
|   |-- recommendation_system_baseline.ipynb
|   `-- ensemble_training.ipynb
|-- scripts/
|   `-- download_movielens_100k.py
`-- src/
    `-- recommender/
        |-- item_cf.py
        |-- metrics.py
        `-- reranking.py
```

## Modeling Pipeline

1. **Item-Based Collaborative Filtering**
   - User-item rating matrix construction
   - User mean centering
   - Item-item cosine similarity
   - Top-K neighbor selection
   - Fallback rules for sparse users/items

2. **Model-Based Collaborative Filtering**
   - SVD, SVD++, and NMF experiments
   - Manual latent factor model implementation
   - Integrated hybrid model with baseline bias, implicit feedback, and item-item neighborhood signal

3. **Weighted Ensemble**
   - Blends item-based local similarity and model-based latent factors
   - Reduces extreme prediction errors from single models
   - Selected as the strongest rating-prediction model in the presentation-level comparison

4. **Long-Tail Reranking**
   - Adds a small inverse-popularity bonus to less frequently rated movies
   - Addresses popularity concentration in the weighted ensemble
   - Evaluates whether coverage gains are meaningful or caused by noisy sparse-item recommendations

## Main Results

### Rating prediction comparison

| Model | RMSE | MAE | Kendall Tau |
| --- | ---: | ---: | ---: |
| Item-based neighborhood CF | 0.9559 | 0.7468 | 0.2965 |
| Model-based CF | 0.9580 | 0.7644 | **0.3961** |
| Weighted hybrid ensemble | **0.9423** | **0.7447** | 0.3177 |

The weighted ensemble achieved the best RMSE and MAE because it combines the local similarity signal from Item-CF with the latent preference signal from model-based CF.

### Notebook validation results

| Experiment | Best setting | Validation RMSE |
| --- | --- | ---: |
| Basic Item-CF | zero-filled similarity | 1.0106 |
| Mean-centered Item-CF | user mean centering | 0.9474 |
| Mean-centered Item-CF Top-K | `K=30` | 0.9305 |
| SVD++ | Surprise SVD++ | 0.9173 |
| Weighted hybrid | Item-CF 0.40 + SVD++ 0.60 | **0.9086** |
| Manual integrated weighted ensemble | Item-CF 0.35 + Integrated Hybrid 0.65 | **0.9094** |

### Long-tail reranking result

Long-tail reranking was used as the final improvement direction because a recommender with strong RMSE can still over-focus on popular movies. A small long-tail bonus increased catalog coverage while keeping RMSE degradation limited.

| Lambda | RMSE | MAE | Coverage@10 | Coverage@20 | Coverage@30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.9419 | 0.7454 | 0.3282 | 0.4227 | 0.4768 |
| 0.02 | 0.9425 | **0.7450** | 0.3312 | 0.4221 | 0.4768 |
| 0.10 | 0.9471 | 0.7450 | 0.3430 | 0.4370 | 0.4768 |
| 0.50 | 1.0176 | 0.7822 | **0.3615** | **0.4483** | 0.4768 |

The project does not treat higher catalog coverage as automatically better. Excessively high coverage can indicate almost random recommendation behavior, especially when many sparse movies with weak rating evidence enter the top-N list. For that reason, the analysis compares standard coverage with min-count filtered coverage and uses long-tail reranking as a controlled adjustment instead of blindly maximizing coverage.

## Reusable Source Code

The `src/recommender` package contains reusable versions of the main notebook logic:

- `item_cf.py`: mean-centered top-K Item-CF artifacts and prediction
- `metrics.py`: RMSE, MAE, and catalog coverage
- `reranking.py`: long-tail bonus creation and reranking

Example:

```python
from recommender import build_item_cf, predict_item_cf

artifacts = build_item_cf(train_ratings, k_neighbors=30)
prediction = predict_item_cf(user_id=1, item_id=50, artifacts=artifacts)
```

## How To Run

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Download MovieLens 100K.

```bash
python scripts/download_movielens_100k.py
```

3. Open the notebooks in Jupyter or Colab.

```bash
jupyter notebook notebooks/recommendation_system_baseline.ipynb
jupyter notebook notebooks/ensemble_training.ipynb
```

The original notebooks were written in Google Colab, so paths such as `/content/drive/MyDrive/movie/ua.base` may need to be changed to `data/raw/ml-100k/ua.base` when running locally.

## Notes

- Raw MovieLens data is not committed to this repository. See `data/README.md`.
- The PDF presentation is stored at `docs/presentation.pdf`.
- The notebooks preserve the full experimental process, while `src/` extracts the reusable logic for portfolio review.
