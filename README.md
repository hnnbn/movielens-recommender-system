# MovieLens Recommendation System

MovieLens 100K 데이터를 활용해 영화 평점 예측과 추천 다양성의 균형을 탐색한 추천시스템 프로젝트입니다. 아이템 기반 이웃 모델, 모델 기반 협업 필터링, 앙상블 모델을 구현하고, 인기 영화 쏠림을 완화하기 위해 long-tail 가중치와 log penalty 계열의 보정 전략을 실험했습니다.

## Project Overview

- **Dataset**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) ratings
- **Task**: user-item rating prediction and top-N recommendation
- **Main metric**: RMSE, MAE, Kendall Tau, Catalog Coverage, Genre Diversity
- **Core approach**: Item-based CF + model-based CF + weighted ensemble
- **Key improvement**: long-tail item weighting to improve catalog coverage with limited accuracy loss

## Repository Structure

```text
movie-recommender-portfolio/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ README.md
├─ docs/
│  ├─ presentation.pdf
│  └─ project-summary.md
├─ notebooks/
│  ├─ README.md
│  ├─ recommendation_system_baseline.ipynb
│  └─ ensemble_training.ipynb
└─ scripts/
   └─ download_movielens_100k.py
```

## Modeling Pipeline

1. **Item-Based Collaborative Filtering**
   - User-item rating matrix construction
   - User mean centering
   - Item-item cosine similarity
   - Top-K neighbor selection
   - Cold-start and sparse-neighbor fallback with global/user/movie mean

2. **Model-Based Collaborative Filtering**
   - SVD, SVD++, NMF experiments
   - Manual latent factor model implementation
   - Integrated hybrid model with baseline bias, implicit feedback, and item-item neighborhood signal

3. **Ensemble**
   - Weighted hybrid between item-based CF and model-based CF
   - Bagging experiments
   - Boosting-style sequential ensemble experiments
   - Cascade hybrid reranking

4. **Long-Tail Improvement**
   - Movie popularity-based long-tail bonus
   - Lambda search to compare RMSE/MAE and catalog coverage trade-off
   - Coverage improvement analysis for `K=10`, `K=20`, and `K=30`

## Main Results

### Final comparison from presentation

| Model | RMSE | MAE | Kendall Tau |
| --- | ---: | ---: | ---: |
| Item-based neighborhood CF | 0.9559 | 0.7468 | 0.2965 |
| Model-based CF | 0.9580 | 0.7644 | 0.3961 |
| Weighted hybrid ensemble | **0.9423** | **0.7447** | 0.3177 |

### Validation results from notebooks

| Experiment | Best setting | Validation RMSE |
| --- | --- | ---: |
| Basic Item-CF | zero-filled similarity | 1.0106 |
| Mean-centered Item-CF | user mean centering | 0.9474 |
| Mean-centered Item-CF Top-K | `K=30` | 0.9305 |
| SVD++ | Surprise SVD++ | 0.9173 |
| Weighted hybrid | Item-CF 0.40 + SVD++ 0.60 | **0.9086** |
| Manual integrated weighted ensemble | Item-CF 0.35 + Integrated Hybrid 0.65 | **0.9094** |

### Catalog coverage trade-off

Long-tail weighted ensemble improved recommendation coverage while keeping the RMSE increase relatively small.

| Lambda | RMSE | MAE | Coverage@10 | Coverage@20 | Coverage@30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.9419 | 0.7454 | 0.3282 | 0.4227 | 0.4768 |
| 0.02 | 0.9425 | 0.7450 | 0.3312 | 0.4221 | 0.4768 |
| 0.10 | 0.9471 | 0.7450 | 0.3430 | 0.4370 | 0.4768 |
| 0.50 | 1.0176 | 0.7822 | 0.3615 | 0.4483 | 0.4768 |

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
- The notebooks preserve the full experimental process, including intermediate model comparisons and final evaluation tables.
