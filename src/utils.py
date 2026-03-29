import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor


def get_lightgbm_model(y_train: np.ndarray):
    if np.issubdtype(y_train.dtype, np.floating):
        task = "regression"
    else:
        task = "classification"

    num_tasks = 1 if y_train.ndim == 1 else y_train.shape[1]

    if task == "classification":
        model = get_single_task_lgbm(task, is_unbalance=True)
        if num_tasks > 1:
            model = MultiOutputClassifier(model)
    else:
        model = get_single_task_lgbm(task, objective="mae")
        if num_tasks > 1:
            model = MultiOutputRegressor(model)

    return task, model


def get_single_task_lgbm(task: str, **kwargs) -> LGBMClassifier | LGBMRegressor:
    defaults = dict(n_estimators=100, n_jobs=-1, random_state=0, verbose=-1)
    defaults.update(kwargs)
    if task == "classification":
        return LGBMClassifier(**defaults)
    return LGBMRegressor(**defaults)
