# Project Summary

## Problem

Movie recommendation models often face a trade-off between rating prediction accuracy and recommendation diversity. Models that optimize only RMSE can over-recommend popular movies, while models that recommend many long-tail movies may suffer from unreliable predictions because sparse items have limited rating history.

This project explores that trade-off on MovieLens 100K by comparing item-based neighborhood CF, model-based CF, and ensemble approaches.

## My Implementation Focus

- Built item-based neighborhood CF with user mean centering and top-K item similarity.
- Compared model-based methods such as SVD, SVD++, NMF, and manually implemented latent factor variants.
- Combined neighborhood and model-based predictions through weighted hybrid ensembles.
- Tested bagging, boosting-style sequential ensembles, and cascade reranking.
- Added popularity-aware long-tail weighting to improve catalog coverage.

## Key Decisions

### Mean Centering

Users have different rating tendencies. Some users rate generously while others rate conservatively. User mean centering reduces that personal scale difference before computing item similarity.

### Top-K Similarity

Using all similar items can introduce noisy neighbors. The validation experiment showed `K=30` as the best item-CF setting in the notebook.

### Weighted Hybrid

Item-CF captures local similarity between movies, while model-based CF captures latent user-item patterns. Weighted hybrid models reduced extreme errors by blending both views.

### Long-Tail Weighting

The ensemble model had better prediction accuracy but still showed popularity concentration. A long-tail bonus based on item popularity was added to make less popular movies more likely to appear in the recommendation list.

## Result Highlights

- Weighted hybrid ensemble achieved the best RMSE in the presentation-level comparison.
- Item-based CF showed strong diversity and high catalog coverage, but part of that coverage came from sparse items with weak rating evidence.
- Model-based CF showed stronger ranking quality but lower diversity and coverage.
- Long-tail weighting increased catalog coverage with a controlled RMSE trade-off.

## Lessons Learned

- Accuracy alone is not enough for recommender-system evaluation.
- Catalog coverage should be interpreted carefully because high coverage can come from unreliable sparse-item recommendations.
- Ensemble models are effective when component models have complementary strengths.
- A small long-tail bonus can make recommendations more diverse, but aggressive weighting quickly hurts rating prediction accuracy.
