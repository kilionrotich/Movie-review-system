"""
Evaluation metrics for the Movie Recommendation System.

Works with both surprise models and the built-in fallback model by only
depending on the common `model.test(...)` prediction interface.
"""

import numpy as np


def rmse(predictions: list) -> float:
    """Compute RMSE from prediction objects exposing est and r_ui."""
    if not predictions:
        return 0.0
    errors = np.array([pred.est - pred.r_ui for pred in predictions], dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(predictions: list) -> float:
    """Compute MAE from prediction objects exposing est and r_ui."""
    if not predictions:
        return 0.0
    errors = np.array([abs(pred.est - pred.r_ui) for pred in predictions], dtype=float)
    return float(errors.mean())


def precision_at_k(
    model,
    testset: list,
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """Compute mean Precision@K across all users in the test set."""
    predictions = model.test(testset)
    user_preds: dict[str, list] = {}
    for pred in predictions:
        user_preds.setdefault(pred.uid, []).append((pred.est, pred.r_ui))

    precisions = []
    for preds in user_preds.values():
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]
        n_relevant = sum(1 for _, r_ui in top_k if r_ui >= threshold)
        precisions.append(n_relevant / k)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    model,
    testset: list,
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """Compute mean Recall@K across all users in the test set."""
    predictions = model.test(testset)
    user_preds: dict[str, list] = {}
    for pred in predictions:
        user_preds.setdefault(pred.uid, []).append((pred.est, pred.r_ui))

    recalls = []
    for preds in user_preds.values():
        n_relevant_total = sum(1 for _, r_ui in preds if r_ui >= threshold)
        if n_relevant_total == 0:
            continue
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]
        n_relevant_in_k = sum(1 for _, r_ui in top_k if r_ui >= threshold)
        recalls.append(n_relevant_in_k / n_relevant_total)
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_all(
    model,
    testset: list,
    k_values: list[int] | None = None,
    threshold: float = 3.5,
) -> dict:
    """Run the full evaluation suite."""
    if k_values is None:
        k_values = [5, 10, 20]

    predictions = model.test(testset)
    metrics = {
        "rmse": rmse(predictions),
        "mae": mae(predictions),
    }
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(model, testset, k=k, threshold=threshold)
        metrics[f"recall@{k}"] = recall_at_k(model, testset, k=k, threshold=threshold)
    return metrics
