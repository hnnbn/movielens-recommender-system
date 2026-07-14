# Project Summary

## Problem

Movie recommendation models have a practical trade-off between rating accuracy and recommendation diversity. A model that optimizes only RMSE may repeatedly recommend popular movies, while a model with very high catalog coverage may include sparse movies whose predicted relevance is unreliable.

This project explores that trade-off on MovieLens 100K by comparing item-based neighborhood CF, model-based CF, weighted ensembles, and long-tail reranking.

## My Implementation Focus

- Built item-based neighborhood CF with user mean centering and top-K item similarity.
- Compared model-based methods such as SVD, SVD++, NMF, and manually implemented latent factor variants.
- Combined neighborhood and model-based predictions through weighted hybrid ensembles.
- Tested bagging, boosting-style sequential ensembles, and cascade reranking.
- Added popularity-aware long-tail reranking to improve catalog coverage without blindly maximizing it.

## Key Decisions

### Mean Centering

Users have different rating tendencies. Some users rate generously while others rate conservatively. User mean centering reduces that personal scale difference before computing item similarity.

### Top-K Similarity

Using all similar items can introduce noisy neighbors. The validation experiment showed `K=30` as the best item-CF setting in the notebook.

### Weighted Hybrid

Item-CF captures local similarity between movies, while model-based CF captures latent user-item patterns. Weighted hybrid models reduced extreme errors by blending both views.

### Catalog Coverage Interpretation

High catalog coverage is not always a good signal. In the item-based model, coverage was very high because sparse long-tail movies often appeared in recommendation lists. That can look diverse, but if the movies have too few ratings, the recommendation may become close to random.

To avoid over-claiming this result, the project compares normal catalog coverage with min-count filtered coverage. This checks whether coverage remains meaningful after removing movies with weak rating evidence.

### Long-Tail Reranking

The weighted ensemble had the best rating accuracy, but it still inherited some popularity concentration from the model-based component. Long-tail reranking adds a small inverse-popularity bonus after prediction, so less popular movies can move up without fully overriding the original relevance score.

This became the main improvement direction because it handles a realistic recommender-system issue: improving item exposure while keeping RMSE degradation controlled.

## Result Highlights

- Weighted hybrid ensemble achieved the best RMSE and MAE in the presentation-level comparison.
- Model-based CF showed the strongest Kendall Tau, meaning it was better at ranking relative preferences.
- Item-based CF showed high diversity and high raw catalog coverage, but some of that coverage came from sparse items.
- Min-count filtering showed that extremely high coverage can be inflated by low-evidence items.
- Long-tail reranking improved coverage with a small, measurable accuracy trade-off.

## Main Takeaway

The final recommendation strategy is not simply the model with the lowest RMSE or the model with the highest coverage. The strongest solution is a weighted ensemble for stable rating prediction, followed by controlled long-tail reranking to improve catalog exposure without making recommendations behave randomly.

## Lessons Learned

- Accuracy alone is not enough for recommender-system evaluation.
- Catalog coverage should be interpreted carefully because very high coverage can come from unreliable sparse-item recommendations.
- Ensemble models are effective when component models have complementary strengths.
- Long-tail reranking is useful when it is treated as a controlled trade-off rather than a goal to maximize at any cost.
