"""
Content-based recommendation utilities.

Uses scikit-learn TF-IDF when available and falls back to a pure-Python
TF-IDF representation when compiled wheels are unavailable.
"""

import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAVE_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    HAVE_SKLEARN = False


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SimpleTfidfVectorizer:
    idf: dict[str, float]


@dataclass
class SimpleTfidfMatrix:
    doc_vectors: list[dict[str, float]]
    norms: list[float]
    vocab_size: int

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.doc_vectors), self.vocab_size)

    def __getitem__(self, index: int) -> dict[str, float]:
        return self.doc_vectors[index]


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _build_simple_tfidf(docs: list[str], min_df: int = 2) -> tuple[SimpleTfidfVectorizer, SimpleTfidfMatrix]:
    tokenized_docs = [_tokenize(doc) for doc in docs]
    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    n_docs = len(tokenized_docs)
    terms = {term for term, freq in doc_freq.items() if freq >= min_df}
    if not terms:
        terms = {term for tokens in tokenized_docs for term in tokens}

    idf = {
        term: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        for term in sorted(terms)
    }

    doc_vectors: list[dict[str, float]] = []
    norms: list[float] = []
    for tokens in tokenized_docs:
        counts = Counter(token for token in tokens if token in idf)
        total = sum(counts.values())
        if total == 0:
            doc_vectors.append({})
            norms.append(0.0)
            continue

        vector = {
            term: (count / total) * idf[term]
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        doc_vectors.append(vector)
        norms.append(norm)

    return SimpleTfidfVectorizer(idf=idf), SimpleTfidfMatrix(
        doc_vectors=doc_vectors,
        norms=norms,
        vocab_size=len(idf),
    )


def _cosine_from_sparse(
    query_vector: dict[str, float],
    query_norm: float,
    matrix: SimpleTfidfMatrix,
) -> np.ndarray:
    scores = np.zeros(len(matrix.doc_vectors), dtype=float)
    if query_norm == 0.0:
        return scores

    for index, doc_vector in enumerate(matrix.doc_vectors):
        doc_norm = matrix.norms[index]
        if doc_norm == 0.0:
            continue
        dot = sum(weight * doc_vector.get(term, 0.0) for term, weight in query_vector.items())
        scores[index] = dot / (query_norm * doc_norm)
    return scores


def build_tfidf_matrix(
    content_df: pd.DataFrame,
    soup_col: str = "soup",
) -> tuple:
    """Build TF-IDF matrix from the soup text column."""
    if HAVE_SKLEARN:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(content_df[soup_col].fillna(""))
        idx_to_movie = content_df["movieId"].to_dict()
        return vectorizer, tfidf_matrix, idx_to_movie

    vectorizer, tfidf_matrix = _build_simple_tfidf(content_df[soup_col].fillna("").tolist())
    idx_to_movie = content_df["movieId"].to_dict()
    return vectorizer, tfidf_matrix, idx_to_movie


def get_similar_movies(
    movie_id: int,
    content_df: pd.DataFrame,
    tfidf_matrix,
    idx_to_movie: dict,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top-N most similar movies to a given movie."""
    movie_to_idx = {v: k for k, v in idx_to_movie.items()}
    if movie_id not in movie_to_idx:
        return pd.DataFrame(columns=["movieId", "title", "similarity_score"])

    idx = movie_to_idx[movie_id]
    if HAVE_SKLEARN:
        query_vec = tfidf_matrix[idx]
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        query_vec = tfidf_matrix[idx]
        sim_scores = _cosine_from_sparse(query_vec, tfidf_matrix.norms[idx], tfidf_matrix)

    sim_scores[idx] = -1.0
    top_indices = np.argsort(sim_scores)[::-1][:n]
    result_ids = [idx_to_movie[i] for i in top_indices]
    result_sims = [float(sim_scores[i]) for i in top_indices]

    title_col = "clean_title" if "clean_title" in content_df.columns else "title"
    id_to_title = content_df.set_index("movieId")[title_col].to_dict()
    return pd.DataFrame({
        "movieId": result_ids,
        "title": [id_to_title.get(mid, "Unknown") for mid in result_ids],
        "similarity_score": result_sims,
    })


def get_profile_recommendations(
    liked_movie_ids: list[int],
    content_df: pd.DataFrame,
    tfidf_matrix,
    idx_to_movie: dict,
    n: int = 10,
    exclude_ids: set | None = None,
) -> pd.DataFrame:
    """Recommend movies based on a user's profile of liked titles."""
    movie_to_idx = {v: k for k, v in idx_to_movie.items()}
    indices = [movie_to_idx[mid] for mid in liked_movie_ids if mid in movie_to_idx]
    if not indices:
        return pd.DataFrame(columns=["movieId", "title", "similarity_score"])

    if HAVE_SKLEARN:
        user_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))
        sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    else:
        aggregate = Counter()
        for index in indices:
            aggregate.update(tfidf_matrix[index])
        user_vector = {term: weight / len(indices) for term, weight in aggregate.items()}
        user_norm = math.sqrt(sum(weight * weight for weight in user_vector.values()))
        sim_scores = _cosine_from_sparse(user_vector, user_norm, tfidf_matrix)

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
    return pd.DataFrame({
        "movieId": result_ids,
        "title": [id_to_title.get(mid, "Unknown") for mid in result_ids],
        "similarity_score": result_sims,
    })


def save_cbf_model(
    vectorizer,
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
