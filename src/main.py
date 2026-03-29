import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from skfp.datasets.moleculeace import (
    load_moleculeace_benchmark,
    load_moleculeace_splits,
)
from skfp.datasets.moleculenet import load_moleculenet_benchmark, load_ogb_splits
from skfp.datasets.tdc import load_tdc_benchmark, load_tdc_splits
from skfp.fingerprints import RDKit2DDescriptorsFingerprint
from skfp.metrics import (
    extract_pos_proba,
    multioutput_auroc_score,
    multioutput_mean_absolute_error,
)
from skfp.preprocessing import MolFromSmilesTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer

from src.feature_selection import (
    MultioutputBoruta,
    MultioutputHSICLasso,
    MultioutputPermutationImportance,
    MultioutputRFECV,
    MultioutputSAGE,
    MultioutputSelectFromModelL1,
    MultioutputSelectPercentile,
    MultioutputSHAP,
    MultioutputShapleyEffects,
    SmartCorrelationSelector,
)
from src.utils import get_lightgbm_model


def split_dataset(
    mols: list[Mol], y: np.ndarray, splits: dict[str, list[str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "valid" in splits:
        train_idxs = splits["train"] + splits["valid"]
    else:
        train_idxs = splits["train"]

    test_idxs = splits["test"]

    mols = np.array(mols)

    mols_train = mols[train_idxs]
    mols_test = mols[test_idxs]
    y_train = y[train_idxs]
    y_test = y[test_idxs]

    return mols_train, mols_test, y_train, y_test


def get_feature_selectors() -> list[tuple[str, Callable]]:
    def _none(task: str) -> None:
        return None

    def _variance_threshold(task: str) -> VarianceThreshold:
        return VarianceThreshold(threshold=0.1)

    def _f_test_80(task: str) -> MultioutputSelectPercentile:
        score_func = f_classif if task == "classification" else f_regression
        return MultioutputSelectPercentile(score_func, percentile=80)

    def _mutual_info_80(task: str) -> MultioutputSelectPercentile:
        score_func = (
            mutual_info_classif if task == "classification" else mutual_info_regression
        )
        return MultioutputSelectPercentile(score_func, percentile=80)

    def _correlation_threshold(task: str) -> SmartCorrelationSelector:
        return SmartCorrelationSelector(threshold=0.95)

    def _boruta(task: str) -> MultioutputBoruta:
        return MultioutputBoruta(task=task)

    def _hsic_lasso(task: str) -> MultioutputHSICLasso:
        return MultioutputHSICLasso(task=task)

    def _permutation_importance(task: str) -> MultioutputPermutationImportance:
        return MultioutputPermutationImportance(task=task)

    def _rfecv(task: str) -> MultioutputRFECV:
        return MultioutputRFECV(task=task)

    def _select_from_model_l1(task: str) -> MultioutputSelectFromModelL1:
        return MultioutputSelectFromModelL1(task=task)

    def _shap(task: str) -> MultioutputSHAP:
        return MultioutputSHAP(task=task, percentile=80)

    def _sage(task: str) -> MultioutputSAGE:
        return MultioutputSAGE(task=task, percentile=80)

    def _shapley_effects(task: str) -> MultioutputShapleyEffects:
        return MultioutputShapleyEffects(task=task, percentile=80)

    return [
        # ("none", _none),
        # ("variance_threshold", _variance_threshold),
        # ("f_test_80", _f_test_80),
        # ("mutual_info_80", _mutual_info_80),
        # ("correlation_threshold", _correlation_threshold),
        # ("boruta", _boruta),
        # ("hsic_lasso", _hsic_lasso),
        # ("permutation_importance", _permutation_importance),
        # ("rfecv", _rfecv),
        # ("select_from_model_l1", _select_from_model_l1),
        # ("shap", _shap),
        # ("sage", _sage),
        ("shapley_effects", _shapley_effects),
    ]


def train_and_eval(
    mols_train: np.ndarray,
    mols_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selector: TransformerMixin | None,
) -> tuple[str, float]:
    fp = RDKit2DDescriptorsFingerprint(n_jobs=-1)
    X_train = fp.transform(mols_train)
    X_test = fp.transform(mols_test)

    # always remove constant features, LightGBM ignores them anyway
    var_thresh = VarianceThreshold()
    X_train = var_thresh.fit_transform(X_train)
    X_test = var_thresh.transform(X_test)

    y_train[np.isnan(y_train)] = 0

    task, model = get_lightgbm_model(y_train)

    if selector is not None:
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        selector.fit(X_train_imputed, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

    model.fit(X_train, y_train)

    if task == "regression":
        y_pred = model.predict(X_test)
        score = multioutput_mean_absolute_error(y_test, y_pred)
    else:
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba = extract_pos_proba(y_pred_proba)
        score = multioutput_auroc_score(y_test, y_pred_proba)

    return task, score


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
        module=r"sklearn\.feature_selection\._univariate_selection",
    )

    results_dir = Path().parent / "results"
    results_dir.mkdir(exist_ok=True)

    mol_from_smiles = MolFromSmilesTransformer(suppress_warnings=True)

    benchmarks = [
        ("MoleculeNet", load_moleculenet_benchmark, load_ogb_splits),
        ("MoleculeACE", load_moleculeace_benchmark, load_moleculeace_splits),
        ("TDC", load_tdc_benchmark, load_tdc_splits),
    ]

    for fs_name, get_selector in get_feature_selectors():
        print(f"Feature selection: {fs_name}")

        for benchmark_name, benchmark_loader, splits_loader in benchmarks:
            print(f"\tBenchmark: {benchmark_name}")

            # reuse already computed reuslts if available
            results_path = results_dir / f"{benchmark_name.lower()}_{fs_name}.csv"
            if results_path.exists():
                existing_df = pd.read_csv(results_path)
                existing_datasets = set(existing_df["dataset"])
                benchmark_scores = existing_df.to_dict("records")
            else:
                existing_datasets = set()
                benchmark_scores = []

            for dataset_name, smiles, y in benchmark_loader():
                if len(smiles) < 500 or len(smiles) > 50000:
                    continue

                if dataset_name in existing_datasets:
                    print(f"\t\t{dataset_name} (cached)")
                    continue
                else:
                    print(f"\t\t{dataset_name}")

                # TODO: bring this back later, for now takes too long
                if dataset_name == "ToxCast":
                    continue

                mols = mol_from_smiles.transform(smiles)

                splits = splits_loader(dataset_name, as_dict=True)
                mols_train, mols_test, y_train, y_test = split_dataset(mols, y, splits)

                task = (
                    "regression"
                    if np.issubdtype(y_train.dtype, np.floating)
                    else "classification"
                )
                selector = get_selector(task)

                task, score = train_and_eval(
                    mols_train, mols_test, y_train, y_test, selector
                )

                benchmark_scores.append(
                    {
                        "dataset": dataset_name,
                        "task": task,
                        "score": score,
                    }
                )

                df = pd.DataFrame(benchmark_scores)
                df.to_csv(results_path, index=False)
