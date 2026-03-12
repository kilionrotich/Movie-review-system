"""
Collaborative filtering utilities.

Uses scikit-surprise when available and falls back to a lightweight
bias-based recommender for environments where compiled dependencies are
not installable, such as Python 3.14 without a C toolchain.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split, cross_validate

    HAVE_SURPRISE = True
except Exception:
    SVD = Any
    Dataset = Any
    Reader = Any
    accuracy = None
    train_test_split = None
    cross_validate = None
    HAVE_SURPRISE = False


MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cf_model.pkl"
)


@dataclass
class Prediction:
    uid: Any
    iid: Any
    r_ui: float
    est: float


class BaselineCFModel:
    """Simple user/item bias model used when scikit-surprise is unavailable."""

    def __init__(
        self,
        global_mean: float,
        user_bias: dict[int, float],
        item_bias: dict[int, float],
    ) -> None:
        self.global_mean = float(global_mean)
        self.user_bias = user_bias
        self.item_bias = item_bias

    @classmethod
    def fit_from_ratings(cls, ratings: pd.DataFrame) -> "BaselineCFModel":
        global_mean = float(ratings["rating"].mean())
        user_stats = ratings.groupby("userId")["rating"].agg(["mean", "count"])
        item_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])

        user_bias = (
            ((user_stats["mean"] - global_mean) * user_stats["count"] / (user_stats["count"] + 10))
            .astype(float)
            .to_dict()
        )
        item_bias = (
            ((item_stats["mean"] - global_mean) * item_stats["count"] / (item_stats["count"] + 10))
            .astype(float)
            .to_dict()
        )
        return cls(global_mean=global_mean, user_bias=user_bias, item_bias=item_bias)

    def predict(self, user_id: Any, movie_id: Any, r_ui: float = 0.0) -> Prediction:
        est = self.global_mean
        est += self.user_bias.get(int(user_id), 0.0)
        est += self.item_bias.get(int(movie_id), 0.0)
        est = float(np.clip(est, 0.5, 5.0))
        return Prediction(uid=user_id, iid=movie_id, r_ui=float(r_ui), est=est)

    def test(self, testset: list[tuple[Any, Any, float]]) -> list[Prediction]:
        return [self.predict(uid, iid, r_ui) for uid, iid, r_ui in testset]


def _build_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    """Convert a ratings DataFrame to a surprise Dataset."""
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)


def _split_ratings(
    ratings: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings without requiring sklearn."""
    if ratings.empty:
        return ratings.copy(), ratings.copy()

    indices = np.arange(len(ratings))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    test_count = max(1, int(round(len(ratings) * test_size))) if len(ratings) > 1 else 0
    test_count = min(test_count, max(len(ratings) - 1, 0))

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    train_df = ratings.iloc[train_idx].reset_index(drop=True)
    test_df = ratings.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def _prediction_metrics(predictions: list[Prediction]) -> dict[str, float]:
    if not predictions:
        return {"rmse": 0.0, "mae": 0.0}

    errors = np.array([pred.est - pred.r_ui for pred in predictions], dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
    }


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
    Train a collaborative-filtering model.

    Uses SVD via surprise when available, otherwise falls back to a bias-based
    recommender that preserves the same public API.
    """
    if HAVE_SURPRISE:
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

    train_df, test_df = _split_ratings(ratings, test_size=test_size, random_state=random_state)
    model = BaselineCFModel.fit_from_ratings(train_df if not train_df.empty else ratings)
    trainset = {"movie_ids": sorted(ratings["movieId"].astype(int).unique().tolist())}
    testset = list(test_df[["userId", "movieId", "rating"]].itertuples(index=False, name=None))
    return model, trainset, testset


def evaluate_svd(model: SVD, testset: list) -> dict:
    """Compute RMSE and MAE on the test set."""
    if HAVE_SURPRISE:
        predictions = model.test(testset)
        return {
            "rmse": accuracy.rmse(predictions, verbose=False),
            "mae": accuracy.mae(predictions, verbose=False),
        }

    return _prediction_metrics(model.test(testset))


def cross_validate_svd(ratings: pd.DataFrame, n_factors: int = 100, cv: int = 5) -> dict:
    """Run k-fold cross validation and return mean RMSE and MAE."""
    if HAVE_SURPRISE:
        data = _build_surprise_dataset(ratings)
        model = SVD(n_factors=n_factors, random_state=42)
        results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
        return {
            "cv_rmse_mean": float(results["test_rmse"].mean()),
            "cv_rmse_std": float(results["test_rmse"].std()),
            "cv_mae_mean": float(results["test_mae"].mean()),
            "cv_mae_std": float(results["test_mae"].std()),
        }

    fold_metrics = []
    shuffled = ratings.sample(frac=1.0, random_state=42).reset_index(drop=True)
    folds = np.array_split(shuffled, cv)

    for fold_idx in range(cv):
        test_df = folds[fold_idx]
        train_parts = [folds[i] for i in range(cv) if i != fold_idx]
        train_df = pd.concat(train_parts, ignore_index=True) if train_parts else shuffled
        model = BaselineCFModel.fit_from_ratings(train_df)
        testset = list(test_df[["userId", "movieId", "rating"]].itertuples(index=False, name=None))
        fold_metrics.append(_prediction_metrics(model.test(testset)))

    rmse_scores = np.array([m["rmse"] for m in fold_metrics], dtype=float)
    mae_scores = np.array([m["mae"] for m in fold_metrics], dtype=float)
    return {
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
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
    """
    if HAVE_SURPRISE:
        all_movie_ids = {trainset.to_raw_iid(iid) for iid in trainset.all_items()}
    else:
        all_movie_ids = set(trainset.get("movie_ids", []))

    if already_rated:
        candidate_ids = all_movie_ids - already_rated
    else:
        candidate_ids = all_movie_ids

    predictions = [(mid, model.predict(user_id, mid).est) for mid in candidate_ids]
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
    """Persist the trained collaborative-filtering model to disk."""
    if path is None:
        path = os.path.abspath(MODEL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str | None = None) -> SVD:
    """Load a persisted collaborative-filtering model from disk."""
    if path is None:
        path = os.path.abspath(MODEL_PATH)
    with open(path, "rb") as f:
        return pickle.load(f)
