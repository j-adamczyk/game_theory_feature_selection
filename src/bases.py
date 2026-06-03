import numpy as np
import sage
from sklearn.base import BaseEstimator, TransformerMixin


class _MultioutputSelector(BaseEstimator, TransformerMixin):
    """
    Common root for all multi-output feature selectors.

    Stores ``mask_`` after fit and provides ``transform``.
    """

    def _apply_fallback(self, n_features: int):
        if not self.mask_.any():
            self.mask_ = np.ones(n_features, dtype=bool)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]


class _MultioutputUnionSelector(_MultioutputSelector):
    """
    Produces a boolean mask per target and ORs them.

    Subclasses implement ``_select_single(X, y) -> np.ndarray[bool]``.
    """

    def _select_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        if y.ndim == 1:
            self.mask_ = self._select_single(X, y)
        else:
            combined = np.zeros(n_features, dtype=bool)
            for i in range(y.shape[1]):
                combined |= self._select_single(X, y[:, i])
            self.mask_ = combined
        self._apply_fallback(n_features)
        return self


class _MultioutputImportanceSelectorBase(_MultioutputSelector):
    """
    Scores features per target, averages across targets, then builds a mask.

    Subclasses implement ``_importances_single(X, y) -> np.ndarray[float]``
    and ``_make_mask(importances) -> np.ndarray[bool]``.
    """

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _make_mask(self, importances: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim == 1:
            importances = self._importances_single(X, y)
        else:
            all_imp = [self._importances_single(X, y[:, i]) for i in range(y.shape[1])]
            importances = np.mean(all_imp, axis=0)
        self.mask_ = self._make_mask(importances)
        self._apply_fallback(len(importances))
        return self


class _MultioutputSignSelector(_MultioutputImportanceSelectorBase):
    """
    Keeps features with non-negative average importance (removes only those
    that actively hurt performance).
    """

    def _make_mask(self, importances: np.ndarray) -> np.ndarray:
        return importances >= 0


class _MultioutputImportanceSelector(_MultioutputImportanceSelectorBase):
    """
    Scores features per target, averages, then thresholds by percentile.

    Subclasses implement ``_importances_single`` and set ``self.percentile``.
    """

    def _make_mask(self, importances: np.ndarray) -> np.ndarray:
        finite = importances[np.isfinite(importances)]
        threshold = np.percentile(finite, 100 - self.percentile)
        return importances >= threshold


class _MissingnessImputer:
    """
    Imputer for models that natively handle missing values (e.g. LightGBM).

    Instead of marginalising out held-out features by averaging over background
    samples, it sets them to NaN and lets the model route them internally.
    This matches the "train a model that accommodates missingness" approach from
    the SAGE paper: no background dataset is needed and each call is O(batch)
    rather than O(batch * n_background).
    """

    def __init__(self, model, num_features: int):
        self.model = sage.utils.model_conversion(model)
        self.num_groups = num_features

    def __call__(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        x_ = x.copy().astype(float)
        x_[~S] = np.nan
        return self.model(x_)
