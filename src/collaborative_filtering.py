"""
Collaborative Filtering using SVD (matrix factorisation).

Uses the scikit-surprise library for training and prediction.
"""

import os
import pickle
from typing import Optional

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cf_model.pkl"
)


def _build_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    """Convert a ratings DataFrame to a surprise Dataset."""
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)


def train_svd(
    ratings: pd.DataFrame,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Train an SVD collaborative-filtering model.

    Parameters
    ----------
    ratings : pd.DataFrame
        Cleaned ratings with columns [userId, movieId, rating].
    n_factors : int
        Number of latent factors.
    n_epochs : int
        Number of SGD epochs.
    lr_all : float
        Learning rate for all parameters.
    reg_all : float
        Regularisation term for all parameters.
    test_size : float
        Fraction of data held out for evaluation.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (trained SVD model, trainset, testset)
    """
    data = _build_surprise_dataset(ratings)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    model.fit(trainset)
    return model, trainset, testset


def evaluate_svd(model: SVD, testset: list) -> dict:
    """Compute RMSE and MAE on the test set."""
    predictions = model.test(testset)
    return {
        "rmse": accuracy.rmse(predictions, verbose=False),
        "mae": accuracy.mae(predictions, verbose=False),
    }


def cross_validate_svd(ratings: pd.DataFrame, n_factors: int = 100, cv: int = 5) -> dict:
    """Run k-fold cross validation and return mean RMSE and MAE."""
    data = _build_surprise_dataset(ratings)
    model = SVD(n_factors=n_factors, random_state=42)
    results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
    return {
        "cv_rmse_mean": float(results["test_rmse"].mean()),
        "cv_rmse_std": float(results["test_rmse"].std()),
        "cv_mae_mean": float(results["test_mae"].mean()),
        "cv_mae_std": float(results["test_mae"].std()),
    }


def get_top_n_cf(
    model: SVD,
    trainset,
    user_id: int,
    movies: pd.DataFrame,
    n: int = 10,
    already_rated: Optional[set] = None,
) -> pd.DataFrame:
    """
    Return top-N movie recommendations for a user via collaborative filtering.

    Parameters
    ----------
    model : SVD
        Trained SVD model.
    trainset : surprise Trainset
        The training set (used to iterate all known movies).
    user_id : int
        Target user ID.
    movies : pd.DataFrame
        Movies DataFrame with at least [movieId, title].
    n : int
        Number of recommendations.
    already_rated : set, optional
        Movie IDs the user has already rated (excluded from results).

    Returns
    -------
    pd.DataFrame
        Columns: movieId, title, estimated_rating
    """
    all_movie_ids = {trainset.to_raw_iid(iid) for iid in trainset.all_items()}
    if already_rated:
        candidate_ids = all_movie_ids - already_rated
    else:
        candidate_ids = all_movie_ids

    predictions = [
        (mid, model.predict(user_id, mid).est) for mid in candidate_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, _ in predictions[:n]]
    top_est = [est for _, est in predictions[:n]]

    result = movies[movies["movieId"].isin(top_ids)][["movieId", "title"]].copy()
    est_map = dict(zip(top_ids, top_est))
    result["estimated_rating"] = result["movieId"].map(est_map)
    result.sort_values("estimated_rating", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def save_model(model: SVD, path: str | None = None) -> str:
    """Persist the trained SVD model to disk."""
    if path is None:
        path = os.path.abspath(MODEL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str | None = None) -> SVD:
    """Load a persisted SVD model from disk."""
    if path is None:
        path = os.path.abspath(MODEL_PATH)
    with open(path, "rb") as f:
        return pickle.load(f)
