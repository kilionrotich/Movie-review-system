"""
Training script for the Movie Recommendation System.

Run this once to download data, train models, and save artifacts:

    python train.py

The script prints evaluation metrics and saves model artifacts to ./data/.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.recommender import MovieRecommender
from src.eda import run_all_eda
from src.preprocessing import clean_movies, clean_tags


def main() -> None:
    print("=" * 60)
    print("Movie Recommendation System – Training")
    print("=" * 60)

    rec = MovieRecommender()

    print("\n[1/4] Downloading and loading data…")
    from src.data_loader import download_movielens, load_all
    data_dir = download_movielens()
    ratings_raw, movies_raw, tags_raw = load_all(data_dir)
    print(
        f"  Loaded {len(ratings_raw):,} ratings, "
        f"{len(movies_raw):,} movies, "
        f"{len(tags_raw):,} tags."
    )

    print("\n[2/4] Running EDA…")
    from src.preprocessing import clean_movies as cm
    clean_m = cm(movies_raw)
    eda_results = run_all_eda(ratings_raw, clean_m, save=True)
    summary = eda_results.pop("summary")
    print("  Dataset summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    print(f"  EDA plots saved to {os.path.abspath('data/eda_plots')}")

    print("\n[3/4] Training collaborative filtering (SVD) and content-based models…")
    metrics = rec.fit(data_dir=data_dir)
    print("  Evaluation metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n[4/4] Saving model artifacts…")
    state_path = rec.save()
    print(f"  Artifacts saved to {state_path}")

    # Persist metrics for reference
    metrics_path = os.path.abspath("data/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Quick sanity check
    print("\nSanity check – top-10 recommendations for user 1 (CF):")
    try:
        recs = rec.recommend_for_user(1, n=10)
        print(recs[["title", "estimated_rating"]].to_string(index=False))
    except Exception as exc:
        print(f"  Warning: {exc}")

    print("\nSanity check – movies similar to 'Toy Story':")
    try:
        similar = rec.similar_to("Toy Story", n=5)
        print(similar[["title", "similarity_score"]].to_string(index=False))
    except Exception as exc:
        print(f"  Warning: {exc}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
