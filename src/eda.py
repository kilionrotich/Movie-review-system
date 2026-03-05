"""
Exploratory Data Analysis utilities for the Movie Recommendation System.
Generates and saves plots; also returns summary statistics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eda_plots")


def _output_dir() -> str:
    path = os.path.abspath(OUTPUT_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def rating_distribution(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot distribution of rating values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(_output_dir(), "rating_distribution.png"), dpi=100)
    return fig


def ratings_per_user(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot histogram of ratings per user."""
    user_counts = ratings.groupby("userId").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(user_counts, bins=50, color="coral", edgecolor="white", log=True)
    ax.set_title("Ratings per User (log scale)")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("User Count")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(_output_dir(), "ratings_per_user.png"), dpi=100)
    return fig


def ratings_per_movie(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot histogram of ratings per movie."""
    movie_counts = ratings.groupby("movieId").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(movie_counts, bins=50, color="mediumseagreen", edgecolor="white", log=True)
    ax.set_title("Ratings per Movie (log scale)")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Movie Count")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(_output_dir(), "ratings_per_movie.png"), dpi=100)
    return fig


def genre_popularity(movies: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart of movie counts by genre."""
    genre_counts: dict[str, int] = {}
    for genres in movies["genres_list"]:
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    genre_series = pd.Series(genre_counts).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=genre_series.values, y=genre_series.index, hue=genre_series.index,
                ax=ax, palette="viridis", legend=False)
    ax.set_title("Movie Count by Genre")
    ax.set_xlabel("Number of Movies")
    ax.set_ylabel("Genre")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(_output_dir(), "genre_popularity.png"), dpi=100)
    return fig


def ratings_over_time(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Line chart of rating volume over time (monthly)."""
    ts = ratings.set_index("timestamp").resample("ME").size()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts.index, ts.values, color="darkorange")
    ax.set_title("Monthly Rating Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Ratings")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(_output_dir(), "ratings_over_time.png"), dpi=100)
    return fig


def top_rated_movies(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings: int = 50,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of highest mean-rated movies."""
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .query(f"count >= {min_ratings}")
        .nlargest(top_n, "mean")
        .reset_index()
    )
    stats = stats.merge(movies[["movieId", "title"]], on="movieId")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=stats, x="mean", y="title", hue="title", ax=ax, palette="Blues_d",
                legend=False)
    ax.set_title(f"Top {top_n} Highest-Rated Movies (min {min_ratings} ratings)")
    ax.set_xlabel("Mean Rating")
    ax.set_ylabel("")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(_output_dir(), "top_rated_movies.png"), dpi=100)
    return fig


def run_all_eda(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    save: bool = True,
) -> dict[str, object]:
    """
    Run all EDA functions and return figures + summary stats.

    Returns
    -------
    dict
        Keys: figure names → matplotlib Figure objects, plus 'summary' key.
    """
    summary = {
        "num_ratings": len(ratings),
        "num_users": ratings["userId"].nunique(),
        "num_movies": ratings["movieId"].nunique(),
        "mean_rating": round(ratings["rating"].mean(), 4),
        "rating_std": round(ratings["rating"].std(), 4),
        "sparsity": round(
            1 - len(ratings) / (ratings["userId"].nunique() * ratings["movieId"].nunique()),
            4,
        ),
    }
    figs = {
        "rating_distribution": rating_distribution(ratings, save=save),
        "ratings_per_user": ratings_per_user(ratings, save=save),
        "ratings_per_movie": ratings_per_movie(ratings, save=save),
        "ratings_over_time": ratings_over_time(ratings, save=save),
        "top_rated_movies": top_rated_movies(ratings, movies, save=save),
    }
    if "genres_list" in movies.columns:
        figs["genre_popularity"] = genre_popularity(movies, save=save)
    figs["summary"] = summary
    return figs
