"""
Unit tests for the Movie Recommendation System.

These tests use synthetic data so no internet connection is required.
"""

import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    extract_year,
    clean_title,
    clean_movies,
    clean_ratings,
    clean_tags,
    build_content_features,
)
from src.content_based import (
    build_tfidf_matrix,
    get_similar_movies,
    get_profile_recommendations,
)
from src.evaluation import precision_at_k, recall_at_k


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_movies() -> pd.DataFrame:
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "GoldenEye (1995)",
            "Four Rooms (1995)",
            "Get Shorty (1995)",
        ],
        "genres": [
            "Animation|Children|Comedy",
            "Adventure|Children|Fantasy",
            "Action|Adventure|Thriller",
            "Comedy|Thriller",
            "Comedy|Crime|Thriller",
        ],
    })


def make_ratings() -> pd.DataFrame:
    rows = []
    for user in range(1, 6):
        for movie in range(1, 6):
            if (user + movie) % 3 != 0:
                rows.append({
                    "userId": user,
                    "movieId": movie,
                    "rating": float(((user * movie) % 5) + 0.5),
                    "timestamp": pd.Timestamp("2023-01-01"),
                })
    return pd.DataFrame(rows)


def make_tags() -> pd.DataFrame:
    return pd.DataFrame({
        "userId": [1, 1, 2, 2, 3],
        "movieId": [1, 2, 1, 3, 4],
        "tag": ["funny", "kids", "funny", "spy", "dark comedy"],
        "timestamp": pd.Timestamp("2023-01-01"),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessing(unittest.TestCase):

    def test_extract_year(self):
        self.assertEqual(extract_year("Toy Story (1995)"), 1995)
        self.assertIsNone(extract_year("Unknown Movie"))

    def test_clean_title(self):
        self.assertEqual(clean_title("Toy Story (1995)"), "Toy Story")
        self.assertEqual(clean_title("No Year"), "No Year")

    def test_clean_movies(self):
        movies = make_movies()
        clean = clean_movies(movies)
        self.assertIn("year", clean.columns)
        self.assertIn("clean_title", clean.columns)
        self.assertIn("genres_list", clean.columns)
        self.assertEqual(clean.loc[0, "year"], 1995)
        self.assertIsInstance(clean.loc[0, "genres_list"], list)
        self.assertIn("Animation", clean.loc[0, "genres_list"])

    def test_clean_ratings_removes_invalid(self):
        df = make_ratings()
        # inject a bad row
        bad = pd.DataFrame({
            "userId": [99], "movieId": [99], "rating": [6.0],
            "timestamp": [pd.Timestamp("2023-01-01")]
        })
        df = pd.concat([df, bad], ignore_index=True)
        clean = clean_ratings(df)
        self.assertTrue((clean["rating"] >= 0.5).all())
        self.assertTrue((clean["rating"] <= 5.0).all())

    def test_clean_ratings_deduplicates(self):
        df = make_ratings()
        dup = df.iloc[:3].copy()
        df = pd.concat([df, dup], ignore_index=True)
        clean = clean_ratings(df)
        duplicates = clean.duplicated(subset=["userId", "movieId"]).sum()
        self.assertEqual(duplicates, 0)

    def test_clean_tags(self):
        tags = make_tags()
        tags_with_blank = pd.concat([
            tags,
            pd.DataFrame({
                "userId": [9], "movieId": [1], "tag": ["   "],
                "timestamp": [pd.Timestamp("2023-01-01")]
            }),
        ], ignore_index=True)
        clean = clean_tags(tags_with_blank)
        self.assertTrue((clean["tag"].str.len() > 0).all())

    def test_build_content_features(self):
        movies = make_movies()
        tags = make_tags()
        content = build_content_features(movies, tags)
        self.assertIn("soup", content.columns)
        self.assertTrue(content["soup"].notna().all())


# ──────────────────────────────────────────────────────────────────────────────
# Content-Based Filtering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestContentBased(unittest.TestCase):

    def setUp(self):
        movies = make_movies()
        tags = make_tags()
        self.content_df = build_content_features(movies, tags)
        self.vectorizer, self.tfidf_matrix, self.idx_to_movie = build_tfidf_matrix(
            self.content_df
        )

    def test_tfidf_matrix_shape(self):
        n_docs = len(self.content_df)
        self.assertEqual(self.tfidf_matrix.shape[0], n_docs)

    def test_similar_movies_returns_n_results(self):
        movie_id = int(self.content_df.iloc[0]["movieId"])
        similar = get_similar_movies(
            movie_id, self.content_df, self.tfidf_matrix, self.idx_to_movie, n=3
        )
        self.assertLessEqual(len(similar), 3)
        self.assertNotIn(movie_id, similar["movieId"].values)

    def test_similar_movies_unknown_id(self):
        result = get_similar_movies(
            9999, self.content_df, self.tfidf_matrix, self.idx_to_movie, n=3
        )
        self.assertTrue(result.empty)

    def test_profile_recommendations(self):
        liked = [int(self.content_df.iloc[0]["movieId"])]
        recs = get_profile_recommendations(
            liked, self.content_df, self.tfidf_matrix, self.idx_to_movie, n=3
        )
        # liked movie should be excluded
        self.assertNotIn(liked[0], recs["movieId"].values)

    def test_profile_recommendations_empty_liked(self):
        recs = get_profile_recommendations(
            [9999], self.content_df, self.tfidf_matrix, self.idx_to_movie, n=3
        )
        self.assertTrue(recs.empty)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation tests (using mock predictions)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluation(unittest.TestCase):

    def _make_full_testset(self):
        """Build a synthetic testset with known relevance (uid, iid, r_ui, est)."""
        # user u1: top-3 by estimate are i1(5.0,r=4), i2(4.0,r=4), i3(3.0,r=2)
        # Relevant items (r >= 3.5): i1, i2, i5
        return [
            ("u1", "i1", 4.0, 5.0),
            ("u1", "i2", 4.0, 4.0),
            ("u1", "i3", 2.0, 3.0),
            ("u1", "i4", 2.0, 2.0),
            ("u1", "i5", 4.0, 1.0),
        ]

    class _MockModel:
        """Minimal surprise-like model for evaluation tests."""
        class _Pred:
            def __init__(self, uid, iid, r_ui, est):
                self.uid, self.iid, self.r_ui, self.est = uid, iid, r_ui, est

        def __init__(self, full_data):
            self._data = full_data

        def test(self, testset):
            return [self._Pred(uid, iid, r_ui, est) for uid, iid, r_ui, est in self._data]

    def test_precision_at_k(self):
        full = self._make_full_testset()
        testset = [(uid, iid, r_ui) for uid, iid, r_ui, est in full]
        model = self._MockModel(full)
        p = precision_at_k(model, testset, k=3, threshold=3.5)
        # Top-3 by est: i1(5.0,r=4), i2(4.0,r=4), i3(3.0,r=2)
        # Relevant in top-3: i1, i2 → 2 → P@3 = 2/3
        self.assertAlmostEqual(p, 2 / 3, places=5)

    def test_recall_at_k(self):
        full = self._make_full_testset()
        testset = [(uid, iid, r_ui) for uid, iid, r_ui, est in full]
        model = self._MockModel(full)
        r = recall_at_k(model, testset, k=3, threshold=3.5)
        # Total relevant (r >= 3.5): i1, i2, i5 → 3
        # Relevant in top-3 (by est): i1, i2 → 2
        # Recall@3 = 2/3
        self.assertAlmostEqual(r, 2 / 3, places=5)


if __name__ == "__main__":
    unittest.main()
