import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MultioutputSelectPercentile(BaseEstimator, TransformerMixin):
    def __init__(self, score_func, percentile: int = 80):
        self.score_func = score_func
        self.percentile = percentile

    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim == 1:
            self.scores_, _ = self.score_func(X, y)
        else:
            all_scores = []
            for i in range(y.shape[1]):
                scores, _ = self.score_func(X, y[:, i])
                all_scores.append(scores)
            self.scores_ = np.mean(all_scores, axis=0)

        threshold = np.percentile(self.scores_, 100 - self.percentile)
        self.mask_ = self.scores_ >= threshold
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]
