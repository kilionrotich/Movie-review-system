"""
Content-Based Filtering using TF-IDF on genres and user tags.

Recommends movies similar to a query movie based on cosine similarity
of their combined genre + tag feature vectors.
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build_tfidf_matrix(
    content_df: pd.DataFrame,
    soup_col: str = "soup",
) -> tuple:
    """
    Build TF-IDF matrix from the 'soup' text column.

    Parameters
    ----------
    content_df : pd.DataFrame
        DataFrame produced by preprocessing.build_content_features().
    soup_col : str
        Column containing the combined text features.

    Returns
    -------
    tuple
        (TfidfVectorizer, sparse tfidf matrix, index→movieId mapping)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(content_df[soup_col].fillna(""))
    idx_to_movie = content_df["movieId"].to_dict()
    return vectorizer, tfidf_matrix, idx_to_movie


def get_similar_movies(
    movie_id: int,
    content_df: pd.DataFrame,
    tfidf_matrix,
    idx_to_movie: dict,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-N most similar movies to a given movie.

    Parameters
    ----------
    movie_id : int
        Source movie ID.
    content_df : pd.DataFrame
        Feature DataFrame (must include 'movieId' and 'title' / 'clean_title').
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix aligned with content_df rows.
    idx_to_movie : dict
        Mapping from row index to movieId.
    n : int
        Number of similar movies to return.

    Returns
    -------
    pd.DataFrame
        Columns: movieId, title, similarity_score
    """
    movie_to_idx = {v: k for k, v in idx_to_movie.items()}
    if movie_id not in movie_to_idx:
        return pd.DataFrame(columns=["movieId", "title", "similarity_score"])

    idx = movie_to_idx[movie_id]
    query_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Exclude the query movie itself
    sim_scores[idx] = -1.0

    top_indices = np.argsort(sim_scores)[::-1][:n]
    result_ids = [idx_to_movie[i] for i in top_indices]
    result_sims = [float(sim_scores[i]) for i in top_indices]

    title_col = "clean_title" if "clean_title" in content_df.columns else "title"
    id_to_title = content_df.set_index("movieId")[title_col].to_dict()

    result = pd.DataFrame({
        "movieId": result_ids,
        "title": [id_to_title.get(mid, "Unknown") for mid in result_ids],
        "similarity_score": result_sims,
    })
    return result


def get_profile_recommendations(
    liked_movie_ids: list[int],
    content_df: pd.DataFrame,
    tfidf_matrix,
    idx_to_movie: dict,
    n: int = 10,
    exclude_ids: set | None = None,
) -> pd.DataFrame:
    """
    Recommend movies based on a user's profile (list of liked movies).

    The user vector is the mean TF-IDF vector of all liked movies.

    Parameters
    ----------
    liked_movie_ids : list of int
        Movie IDs the user likes.
    content_df : pd.DataFrame
        Content features DataFrame.
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix.
    idx_to_movie : dict
        Row index → movieId.
    n : int
        Number of recommendations.
    exclude_ids : set, optional
        Movie IDs to exclude from results (e.g., already liked).

    Returns
    -------
    pd.DataFrame
        Columns: movieId, title, similarity_score
    """
    movie_to_idx = {v: k for k, v in idx_to_movie.items()}
    indices = [movie_to_idx[mid] for mid in liked_movie_ids if mid in movie_to_idx]

    if not indices:
        return pd.DataFrame(columns=["movieId", "title", "similarity_score"])

    user_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    if exclude_ids is None:
        exclude_ids = set(liked_movie_ids)
    else:
        exclude_ids = exclude_ids | set(liked_movie_ids)

    for mid in exclude_ids:
        if mid in movie_to_idx:
            sim_scores[movie_to_idx[mid]] = -1.0

    top_indices = np.argsort(sim_scores)[::-1][:n]
    result_ids = [idx_to_movie[i] for i in top_indices]
    result_sims = [float(sim_scores[i]) for i in top_indices]

    title_col = "clean_title" if "clean_title" in content_df.columns else "title"
    id_to_title = content_df.set_index("movieId")[title_col].to_dict()

    result = pd.DataFrame({
        "movieId": result_ids,
        "title": [id_to_title.get(mid, "Unknown") for mid in result_ids],
        "similarity_score": result_sims,
    })
    return result


def save_cbf_model(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    idx_to_movie: dict,
    path: str | None = None,
) -> str:
    """Persist the content-based model artifacts."""
    if path is None:
        path = os.path.abspath(os.path.join(MODEL_DIR, "cbf_model.pkl"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "idx_to_movie": idx_to_movie}, f)
    return path


def load_cbf_model(path: str | None = None) -> tuple:
    """Load persisted content-based model artifacts."""
    if path is None:
        path = os.path.abspath(os.path.join(MODEL_DIR, "cbf_model.pkl"))
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["vectorizer"], obj["tfidf_matrix"], obj["idx_to_movie"]
