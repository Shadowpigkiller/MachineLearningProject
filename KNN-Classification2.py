import os
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# keep other warnings visible except the specific CV warning we avoid by design

DATA_DIR = "Data/classification"


def safe_load(path, dtype=float):
    """Load a whitespace-delimited text file robustly."""
    return np.loadtxt(path, dtype=dtype)


def compute_safe_cv_splits(y, preferred_splits=5, min_allowed=2):
    """
    Choose number of CV splits so that each fold can contain at least one sample
    from the smallest class when using StratifiedKFold.
    """
    class_counts = np.bincount(np.asarray(y, dtype=int))
    min_class = class_counts[class_counts > 0].min()
    # At most min_class splits; but want at least min_allowed
    n_splits = max(min_allowed, min(preferred_splits, int(min_class)))
    # If min_class < 2 then StratifiedKFold can't be used; fallback to simple KFold-like behavior
    return n_splits


def build_pipeline(use_pca=True):
    """
    Builds a sklearn Pipeline with placeholders. PCA is included but GridSearch may
    choose to set n_components=None (via param grid) to effectively disable PCA.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("pca", PCA()),  # PCA will be configured by GridSearchCV params
        ("knn", KNeighborsClassifier())
    ]
    return Pipeline(steps)


def param_grid_for_knn():
    """
    Returns a parameter grid that:
    - chooses whether to use PCA (via n_components values)
    - tunes K (n_neighbors), weights, and metric
    """
    # PCA n_components as fraction of explained variance. Include "None" by setting n_components = 0.0 flag later.
    # sklearn PCA does not accept None through param_grid directly; we will use small trick:
    # set 'pca__n_components' to values and if 'pca__n_components' equals 'none' we will replace PCA with 'identity' in the pipeline.
    # But GridSearch can't change estimator types. Simpler: include a very high n_components = 0.9999 which keeps most info.
    # We'll include three fractions; for tiny sample-high-dim datasets, PCA will reduce dimensions a lot.
    grid = {
        # PCA: keep 90%, 95%, 99% variance and also near-1.0 to approximate "no reduction"
        "pca__n_components": [0.90, 0.95, 0.99, 0.9999],
        "knn__n_neighbors": [1, 3, 5, 7, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }
    return grid


def run_dataset(setnum, train_X, test_X, train_y, out_dir="."):
    print(f"\n=== Dataset {setnum} ===")
    n_samples, n_features = train_X.shape
    print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}, Classes: {np.unique(train_y)}")

    # Determine safe CV splits
    cv_splits = compute_safe_cv_splits(train_y, preferred_splits=5, min_allowed=2)
    print(f"Using StratifiedKFold with n_splits={cv_splits}")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    pipe = build_pipeline(use_pca=True)
    grid = param_grid_for_knn()

    # GridSearchCV
    gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=1)

    t0 = time.time()
    gs.fit(train_X, train_y)
    t1 = time.time()

    print(f"GridSearch completed in {t1 - t0:.1f}s")
    print("Best params:", gs.best_params_)
    print("Best CV accuracy: {:.4f}".format(gs.best_score_))

    # Best estimator -> predict test
    preds = gs.predict(test_X)
    probs = None
    if hasattr(gs.best_estimator_["knn"], "predict_proba"):
        probs = gs.predict_proba(test_X)

    # Save predictions
    out_pred_path = os.path.join(out_dir, f"predictions_set{setnum}.csv")
    pd.DataFrame({"prediction": preds}).to_csv(out_pred_path, index=False)
    print(f"Predictions saved to {out_pred_path}")

    return gs.best_estimator_, preds


def main():
    # Load all datasets from the Data/classification folder
    sets = []
    for i in range(1, 5):
        train_data = safe_load(os.path.join(DATA_DIR, f"TrainData{i}.txt"))
        test_data = safe_load(os.path.join(DATA_DIR, f"TestData{i}.txt"))
        train_labels = safe_load(os.path.join(DATA_DIR, f"TrainLabel{i}.txt"), dtype=int)
        sets.append((train_data, test_data, train_labels))

    # Replace the sentinel (1e99) with np.nan - safe as we are using SimpleImputer inside pipeline
    for idx, (train_X, test_X, train_y) in enumerate(sets, start=1):
        train_X[train_X == 1e99] = np.nan
        test_X[test_X == 1e99] = np.nan

    # Run each dataset
    results = {}
    out_dir = "."
    for i, (train_X, test_X, train_y) in enumerate(sets, start=1):
        best_estimator, preds = run_dataset(i, train_X, test_X, train_y, out_dir=out_dir)
        results[i] = {"estimator": best_estimator, "predictions": preds}

    print("\nAll done. Prediction files: predictions_set1.csv ... predictions_set4.csv")


if __name__ == "__main__":
    main()
