"""
High-level Recommender that combines Collaborative Filtering (SVD)
and Content-Based Filtering (TF-IDF).

Provides a unified interface for training, persisting, and querying
recommendations.
"""

import os
import pickle
from typing import Optional

import pandas as pd

from src.data_loader import load_all, download_movielens
from src.preprocessing import clean_ratings, clean_movies, clean_tags, build_content_features
from src.collaborative_filtering import (
    train_svd,
    evaluate_svd,
    get_top_n_cf,
    save_model as save_cf,
    load_model as load_cf,
)
from src.content_based import (
    build_tfidf_matrix,
    get_similar_movies,
    get_profile_recommendations,
    save_cbf_model,
    load_cbf_model,
)
from src.evaluation import evaluate_all


_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_STATE_PATH = os.path.join(_ARTIFACT_DIR, "recommender_state.pkl")


class MovieRecommender:
    """
    End-to-end movie recommender combining CF and CBF approaches.

    Usage
    -----
    >>> rec = MovieRecommender()
    >>> rec.fit()          # downloads data, trains models
    >>> rec.recommend_for_user(42, n=10)
    >>> rec.similar_to("Toy Story (1995)", n=10)
    """

    def __init__(self) -> None:
        self.cf_model = None
        self.cf_trainset = None
        self.cf_testset = None

        self.vectorizer = None
        self.tfidf_matrix = None
        self.idx_to_movie: dict = {}

        self.ratings: Optional[pd.DataFrame] = None
        self.movies: Optional[pd.DataFrame] = None
        self.content_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data_dir: str | None = None,
        n_factors: int = 100,
        n_epochs: int = 20,
    ) -> dict:
        """
        Download data (if needed), preprocess, train both models, and evaluate.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        if data_dir is None:
            data_dir = download_movielens()

        ratings_raw, movies_raw, tags_raw = load_all(data_dir)

        self.ratings = clean_ratings(ratings_raw)
        clean_m = clean_movies(movies_raw)
        clean_t = clean_tags(tags_raw)

        self.movies = clean_m
        self.content_df = build_content_features(movies_raw, tags_raw)

        # Collaborative Filtering
        self.cf_model, self.cf_trainset, self.cf_testset = train_svd(
            self.ratings, n_factors=n_factors, n_epochs=n_epochs
        )

        # Content-Based Filtering
        self.vectorizer, self.tfidf_matrix, self.idx_to_movie = build_tfidf_matrix(
            self.content_df
        )

        metrics = evaluate_all(self.cf_model, self.cf_testset)
        return metrics

    # ------------------------------------------------------------------
    # Collaborative-filtering recommendations
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return top-N CF recommendations for a user.

        Excludes movies the user has already rated.
        """
        if self.cf_model is None or self.ratings is None:
            raise RuntimeError("Call fit() before recommending.")
        already_rated = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        )
        return get_top_n_cf(
            self.cf_model,
            self.cf_trainset,
            user_id,
            self.movies,
            n=n,
            already_rated=already_rated,
        )

    # ------------------------------------------------------------------
    # Content-based recommendations
    # ------------------------------------------------------------------

    def similar_to(self, title_or_id, n: int = 10) -> pd.DataFrame:
        """
        Return movies similar to a given title or movie ID.

        Parameters
        ----------
        title_or_id : str or int
            Movie title (substring match) or exact movieId.
        n : int
            Number of results.
        """
        if self.content_df is None:
            raise RuntimeError("Call fit() before recommending.")
        movie_id = self._resolve_movie_id(title_or_id)
        if movie_id is None:
            return pd.DataFrame(columns=["movieId", "title", "similarity_score"])
        return get_similar_movies(
            movie_id, self.content_df, self.tfidf_matrix, self.idx_to_movie, n=n
        )

    def recommend_from_liked(
        self,
        liked_titles_or_ids: list,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend movies based on a list of movies the user likes (CBF).
        """
        if self.content_df is None:
            raise RuntimeError("Call fit() before recommending.")
        liked_ids = [
            self._resolve_movie_id(t) for t in liked_titles_or_ids
        ]
        liked_ids = [mid for mid in liked_ids if mid is not None]
        return get_profile_recommendations(
            liked_ids,
            self.content_df,
            self.tfidf_matrix,
            self.idx_to_movie,
            n=n,
        )

    # ------------------------------------------------------------------
    # Hybrid recommendations
    # ------------------------------------------------------------------

    def hybrid_recommend(
        self,
        user_id: int | None = None,
        liked_titles_or_ids: list | None = None,
        n: int = 10,
        cf_weight: float = 0.5,
    ) -> pd.DataFrame:
        """
        Blend CF and CBF scores for a richer recommendation list.

        At least one of user_id or liked_titles_or_ids must be supplied.
        """
        if self.cf_model is None or self.content_df is None:
            raise RuntimeError("Call fit() before recommending.")

        cf_recs = pd.DataFrame(columns=["movieId", "estimated_rating"])
        cbf_recs = pd.DataFrame(columns=["movieId", "similarity_score"])

        if user_id is not None:
            cf_recs = self.recommend_for_user(user_id, n=n * 3)

        if liked_titles_or_ids:
            cbf_recs = self.recommend_from_liked(liked_titles_or_ids, n=n * 3)

        if cf_recs.empty and cbf_recs.empty:
            return pd.DataFrame(columns=["movieId", "title", "score"])

        if not cf_recs.empty:
            cf_max = cf_recs["estimated_rating"].max()
            cf_recs = cf_recs.copy()
            cf_recs["cf_score"] = cf_recs["estimated_rating"] / cf_max if cf_max > 0 else 0.0

        if not cbf_recs.empty:
            cbf_max = cbf_recs["similarity_score"].max()
            cbf_recs = cbf_recs.copy()
            cbf_recs["cbf_score"] = cbf_recs["similarity_score"] / cbf_max if cbf_max > 0 else 0.0

        all_ids = set()
        if not cf_recs.empty:
            all_ids.update(cf_recs["movieId"].tolist())
        if not cbf_recs.empty:
            all_ids.update(cbf_recs["movieId"].tolist())

        rows = []
        cf_lookup = {} if cf_recs.empty else cf_recs.set_index("movieId")["cf_score"].to_dict()
        cbf_lookup = {} if cbf_recs.empty else cbf_recs.set_index("movieId")["cbf_score"].to_dict()

        for mid in all_ids:
            cf_s = cf_lookup.get(mid, 0.0)
            cbf_s = cbf_lookup.get(mid, 0.0)
            rows.append({"movieId": mid, "score": cf_weight * cf_s + (1 - cf_weight) * cbf_s})

        hybrid = pd.DataFrame(rows).nlargest(n, "score")
        title_col = "clean_title" if "clean_title" in self.movies.columns else "title"
        id_to_title = self.movies.set_index("movieId")[title_col].to_dict()
        hybrid["title"] = hybrid["movieId"].map(id_to_title)
        hybrid = hybrid[["movieId", "title", "score"]].reset_index(drop=True)
        return hybrid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Persist the full recommender state."""
        if path is None:
            path = os.path.abspath(_STATE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "idx_to_movie": self.idx_to_movie,
            "ratings": self.ratings,
            "movies": self.movies,
            "content_df": self.content_df,
            "cf_trainset": self.cf_trainset,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        save_cf(self.cf_model)
        return path

    def load(self, path: str | None = None) -> None:
        """Load a persisted recommender state."""
        if path is None:
            path = os.path.abspath(_STATE_PATH)
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self.tfidf_matrix = state["tfidf_matrix"]
        self.idx_to_movie = state["idx_to_movie"]
        self.ratings = state["ratings"]
        self.movies = state["movies"]
        self.content_df = state["content_df"]
        self.cf_trainset = state.get("cf_trainset")
        self.cf_model = load_cf()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_movie_id(self, title_or_id) -> Optional[int]:
        """Resolve a title (substring) or int ID to a movieId."""
        if isinstance(title_or_id, int):
            if title_or_id in self.content_df["movieId"].values:
                return title_or_id
            return None
        # String: try exact match on title first, then substring
        title_col = "title" if "title" in self.content_df.columns else "clean_title"
        exact = self.content_df[
            self.content_df[title_col].str.lower() == title_or_id.lower()
        ]
        if not exact.empty:
            return int(exact.iloc[0]["movieId"])
        partial = self.content_df[
            self.content_df[title_col].str.lower().str.contains(
                title_or_id.lower(), regex=False
            )
        ]
        if not partial.empty:
            return int(partial.iloc[0]["movieId"])
        return None

    def search_movies(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """Return movies whose titles contain the query string."""
        if self.movies is None:
            raise RuntimeError("Call fit() before searching.")
        title_col = "clean_title" if "clean_title" in self.movies.columns else "title"
        mask = self.movies[title_col].str.lower().str.contains(
            query.lower(), regex=False
        )
        results = self.movies[mask][["movieId", title_col, "genres"]].head(top_n)
        results = results.rename(columns={title_col: "title"})
        return results.reset_index(drop=True)

    def list_genres(self) -> list[str]:
        """Return a sorted list of all unique genres."""
        if self.movies is None:
            raise RuntimeError("Call fit() before querying genres.")
        genres: set[str] = set()
        for gl in self.movies["genres_list"]:
            genres.update(gl)
        return sorted(genres)

    def movies_by_genre(self, genre: str, top_n: int = 20) -> pd.DataFrame:
        """Return movies that belong to a given genre."""
        if self.movies is None:
            raise RuntimeError("Call fit() before querying by genre.")
        mask = self.movies["genres_list"].apply(lambda gl: genre in gl)
        title_col = "clean_title" if "clean_title" in self.movies.columns else "title"
        results = self.movies[mask][["movieId", title_col, "year"]].head(top_n)
        return results.rename(columns={title_col: "title"}).reset_index(drop=True)
