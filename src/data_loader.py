"""
Data loader for MovieLens dataset.
Downloads and caches the MovieLens Small dataset (100K ratings).
"""

import os
import io
import zipfile
import requests
import pandas as pd


MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _data_dir() -> str:
    """Return the absolute path to the data directory, creating it if needed."""
    path = os.path.abspath(DATA_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def download_movielens(force: bool = False) -> str:
    """
    Download and extract the MovieLens Small dataset.

    If the download fails (e.g., no network access), falls back to generating
    a synthetic dataset with the same schema.

    Parameters
    ----------
    force : bool
        Re-download even if data already exists.

    Returns
    -------
    str
        Path to the extracted dataset directory.
    """
    data_dir = _data_dir()
    ml_dir = os.path.join(data_dir, "ml-latest-small")
    ratings_path = os.path.join(ml_dir, "ratings.csv")

    if not force and os.path.exists(ratings_path):
        return ml_dir

    try:
        print("Downloading MovieLens Small dataset…")
        response = requests.get(MOVIELENS_URL, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(data_dir)
        print(f"Dataset saved to {ml_dir}")
    except Exception as exc:
        print(f"Download failed ({exc}). Generating synthetic dataset instead…")
        _generate_fallback(ml_dir)

    return ml_dir


def _generate_fallback(ml_dir: str) -> None:
    """Generate and save a synthetic MovieLens-like dataset."""
    # Import here to avoid circular dependency at module level
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from generate_sample_data import generate_dataset, save_dataset
    ratings, movies, tags = generate_dataset(n_users=500, seed=42)
    save_dataset(ratings, movies, tags, ml_dir)


def load_ratings(data_dir: str | None = None) -> pd.DataFrame:
    """Load ratings.csv into a DataFrame."""
    if data_dir is None:
        data_dir = download_movielens()
    path = os.path.join(data_dir, "ratings.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies(data_dir: str | None = None) -> pd.DataFrame:
    """Load movies.csv into a DataFrame."""
    if data_dir is None:
        data_dir = download_movielens()
    path = os.path.join(data_dir, "movies.csv")
    return pd.read_csv(path)


def load_tags(data_dir: str | None = None) -> pd.DataFrame:
    """Load tags.csv into a DataFrame."""
    if data_dir is None:
        data_dir = download_movielens()
    path = os.path.join(data_dir, "tags.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_all(data_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three core tables.

    Returns
    -------
    tuple
        (ratings, movies, tags)
    """
    if data_dir is None:
        data_dir = download_movielens()
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    tags = load_tags(data_dir)
    return ratings, movies, tags
