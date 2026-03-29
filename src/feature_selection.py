import numpy as np
import pandas as pd
import sage
import shap
from boruta import BorutaPy
from feature_engine.selection import SmartCorrelatedSelection
from pyHSICLasso import HSICLasso
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold

from src.utils import get_single_task_lgbm


class _MultioutputUnionSelector(BaseEstimator, TransformerMixin):
    """
    Base for selectors that produce a boolean mask per target and OR them.

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

        if not self.mask_.any():
            self.mask_ = np.ones(n_features, dtype=bool)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]


class _MultioutputImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Base for selectors that score features per target, average, then threshold.

    Subclasses implement ``_importances_single(X, y) -> np.ndarray[float]``
    and set ``self.percentile``.
    """

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim == 1:
            importances = self._importances_single(X, y)
        else:
            all_imp = [self._importances_single(X, y[:, i]) for i in range(y.shape[1])]
            importances = np.mean(all_imp, axis=0)

        finite = importances[np.isfinite(importances)]
        threshold = np.percentile(finite, 100 - self.percentile)
        self.mask_ = importances >= threshold
        if not self.mask_.any():
            self.mask_ = np.ones(len(importances), dtype=bool)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]


# concrete feature selectors


class MultioutputSelectPercentile(_MultioutputImportanceSelector):
    """
    Univariate feature selection, e.g. ANOVA F-value, mutual information.
    """

    def __init__(self, score_func, percentile: int = 80):
        self.score_func = score_func
        self.percentile = percentile

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        result = self.score_func(X, y)
        return result[0] if isinstance(result, tuple) else result


class SmartCorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Correlated features remover (unsupervised).
    """

    def __init__(self, threshold: float = 0.9, method: str = "pearson"):
        self.threshold = threshold
        self.method = method

    def fit(self, X: np.ndarray, y=None):
        columns = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=columns)

        selector = SmartCorrelatedSelection(
            method=self.method,
            threshold=self.threshold,
            selection_method="variance",
            missing_values="ignore",
        )
        selector.fit(X_df)
        dropped = set(selector.features_to_drop_)
        self.mask_ = np.array([c not in dropped for c in columns])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]


class MultioutputBoruta(_MultioutputUnionSelector):
    """
    Boruta algorithm.
    """

    def __init__(self, task: str = "classification", max_iter: int = 50):
        self.task = task
        self.max_iter = max_iter

    def _select_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            estimator = RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=0
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100, n_jobs=-1, random_state=0
            )
        boruta = BorutaPy(
            estimator,
            n_estimators="auto",
            random_state=0,
            max_iter=self.max_iter,
            verbose=0,
        )
        boruta.fit(X, y)
        return boruta.support_


class MultioutputHSICLasso(_MultioutputUnionSelector):
    """
    HSIC Lasso.
    """

    def __init__(self, task: str = "classification", num_feat: int = 50):
        self.task = task
        self.num_feat = num_feat

    def _select_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        hsic = HSICLasso()
        num_feat = min(self.num_feat, X.shape[1])
        hsic.input(X, y)
        if self.task == "classification":
            hsic.classification(num_feat=num_feat, n_jobs=-1)
        else:
            hsic.regression(num_feat=num_feat, n_jobs=-1)
        selected = set(hsic.get_index())
        return np.array([i in selected for i in range(X.shape[1])])


class MultioutputRFECV(_MultioutputUnionSelector):
    """
    RFECV.
    """

    def __init__(self, task: str = "classification"):
        self.task = task

    def _select_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=0)

        scoring = (
            "roc_auc" if self.task == "classification" else "neg_mean_absolute_error"
        )
        rfecv = RFECV(
            estimator=get_single_task_lgbm(self.task),
            step=1,
            cv=cv,
            scoring=scoring,
            min_features_to_select=100,
            n_jobs=-1,
        )
        rfecv.fit(X, y)
        return rfecv.support_


class MultioutputSelectFromModelL1(_MultioutputUnionSelector):
    """
    Linear models with L1 regularization.
    """

    def __init__(self, task: str = "classification"):
        self.task = task

    def _select_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            model = LogisticRegressionCV(
                Cs=100,
                l1_ratios=[1],
                solver="saga",
                max_iter=1000,
                random_state=0,
            )
        else:
            model = LassoCV(n_alphas=100, random_state=0, max_iter=1000)
        selector = SelectFromModel(estimator=model)
        selector.fit(X, y)
        return selector.get_support()


class MultioutputPermutationImportance(_MultioutputImportanceSelector):
    """
    Permutation feature importance.
    """

    def __init__(self, task: str = "classification", percentile: int = 80):
        self.task = task
        self.percentile = percentile

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = get_single_task_lgbm(self.task)
        model.fit(X, y)
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=0, n_jobs=-1
        )
        return result.importances_mean


class MultioutputSHAP(_MultioutputImportanceSelector):
    """
    Mean absolute SHAP values.
    """

    def __init__(self, task: str = "classification", percentile: int = 80):
        self.task = task
        self.percentile = percentile

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = get_single_task_lgbm(self.task)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            # binary classification returns [class0, class1]
            shap_values = shap_values[1]
        return np.abs(shap_values).mean(axis=0)


class MultioutputSAGE(_MultioutputImportanceSelector):
    """
    SAGE values.
    """

    def __init__(self, task: str = "classification", percentile: int = 80):
        self.task = task
        self.percentile = percentile

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = get_single_task_lgbm(self.task)
        model.fit(X, y)
        n_bg = min(512, X.shape[0])
        imputer = sage.MarginalImputer(model, X[:n_bg])
        loss = "cross entropy" if self.task == "classification" else "mse"
        estimator = sage.PermutationEstimator(imputer, loss)
        return estimator(X, y).values


class MultioutputShapleyEffects(_MultioutputImportanceSelector):
    """
    Shaply Effects (unsupervised).
    """

    def __init__(self, task: str = "classification", percentile: int = 80):
        self.task = task
        self.percentile = percentile

    def _importances_single(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = get_single_task_lgbm(self.task)
        model.fit(X, y)
        n_bg = min(512, X.shape[0])
        imputer = sage.MarginalImputer(model, X[:n_bg])
        loss = "cross entropy" if self.task == "classification" else "mse"
        estimator = sage.PermutationEstimator(imputer, loss)
        # no Y gives Shapley Effects
        return estimator(X).values
