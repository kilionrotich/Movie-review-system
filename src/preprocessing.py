"""
Data cleaning and preprocessing for the Movie Recommendation System.
"""

import re
import pandas as pd


def extract_year(title: str) -> int | None:
    """Extract the release year embedded in a movie title, e.g. 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return None


def clean_title(title: str) -> str:
    """Remove the trailing year from a movie title."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the movies DataFrame.

    - Extract year from title.
    - Clean title string.
    - Split genres into a list column.
    - Drop rows missing genres.
    """
    df = movies.copy()
    df["year"] = df["title"].apply(extract_year)
    df["clean_title"] = df["title"].apply(clean_title)
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    df = df[df["genres"] != "(no genres listed)"].copy()
    df.drop_duplicates(subset="movieId", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the ratings DataFrame.

    - Drop duplicates (keep last per user–movie pair).
    - Remove ratings outside [0.5, 5.0].
    - Drop rows with nulls.
    """
    df = ratings.copy()
    df.drop_duplicates(subset=["userId", "movieId"], keep="last", inplace=True)
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the tags DataFrame.

    - Lowercase and strip whitespace.
    - Drop blank or null tags.
    - Drop duplicates.
    """
    df = tags.copy()
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    df = df[df["tag"].str.len() > 0].copy()
    df.drop_duplicates(subset=["userId", "movieId", "tag"], inplace=True)
    df.dropna(subset=["tag"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def merge_movie_tags(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all tags per movie and merge with movies DataFrame.

    The resulting 'tag_str' column contains space-separated tags, ready for
    TF-IDF vectorisation.
    """
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"tag": "tag_str"})
    )
    merged = movies.merge(tag_agg, on="movieId", how="left")
    merged["tag_str"] = merged["tag_str"].fillna("")
    return merged


def build_content_features(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame with combined text features for content-based filtering.

    Combines genres and tags into a single 'soup' column.
    """
    clean_m = clean_movies(movies)
    clean_t = clean_tags(tags)
    merged = merge_movie_tags(clean_m, clean_t)
    merged["genres_str"] = merged["genres_list"].apply(" ".join)
    merged["soup"] = merged["genres_str"] + " " + merged["tag_str"]
    merged["soup"] = merged["soup"].str.strip()
    return merged
