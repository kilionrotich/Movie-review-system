"""
Evaluation metrics for the Movie Recommendation System.

Implements:
- RMSE  (Root Mean Squared Error) for rating prediction
- Precision@K and Recall@K for top-N recommendations
"""

import numpy as np
import pandas as pd
from surprise import SVD, accuracy
from surprise.model_selection import train_test_split


def rmse(predictions: list) -> float:
    """Compute RMSE from a list of surprise Prediction namedtuples."""
    return float(accuracy.rmse(predictions, verbose=False))


def mae(predictions: list) -> float:
    """Compute MAE from a list of surprise Prediction namedtuples."""
    return float(accuracy.mae(predictions, verbose=False))


def precision_at_k(
    model: SVD,
    testset: list,
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """
    Compute mean Precision@K across all users in the test set.

    A recommendation is considered *relevant* if the true rating
    is >= threshold.

    Parameters
    ----------
    model : SVD
        Trained surprise model.
    testset : list
        List of (uid, iid, true_rating) tuples.
    k : int
        Cutoff rank.
    threshold : float
        Minimum true rating to be counted as relevant.

    Returns
    -------
    float
        Mean Precision@K.
    """
    predictions = model.test(testset)
    user_preds: dict[str, list] = {}
    for pred in predictions:
        user_preds.setdefault(pred.uid, []).append((pred.est, pred.r_ui))

    precisions = []
    for uid, preds in user_preds.items():
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]
        n_relevant = sum(1 for _, r_ui in top_k if r_ui >= threshold)
        precisions.append(n_relevant / k)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    model: SVD,
    testset: list,
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """
    Compute mean Recall@K across all users in the test set.

    Parameters
    ----------
    model : SVD
        Trained surprise model.
    testset : list
        List of (uid, iid, true_rating) tuples.
    k : int
        Cutoff rank.
    threshold : float
        Minimum true rating to be counted as relevant.

    Returns
    -------
    float
        Mean Recall@K.
    """
    predictions = model.test(testset)
    user_preds: dict[str, list] = {}
    for pred in predictions:
        user_preds.setdefault(pred.uid, []).append((pred.est, pred.r_ui))

    recalls = []
    for uid, preds in user_preds.items():
        n_relevant_total = sum(1 for _, r_ui in preds if r_ui >= threshold)
        if n_relevant_total == 0:
            continue
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]
        n_relevant_in_k = sum(1 for _, r_ui in top_k if r_ui >= threshold)
        recalls.append(n_relevant_in_k / n_relevant_total)
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_all(
    model: SVD,
    testset: list,
    k_values: list[int] | None = None,
    threshold: float = 3.5,
) -> dict:
    """
    Run the full evaluation suite.

    Returns a dict with RMSE, MAE, and Precision@K / Recall@K for each k.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    metrics = {
        "rmse": rmse(model.test(testset)),
        "mae": mae(model.test(testset)),
    }
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(model, testset, k=k, threshold=threshold)
        metrics[f"recall@{k}"] = recall_at_k(model, testset, k=k, threshold=threshold)
    return metrics
