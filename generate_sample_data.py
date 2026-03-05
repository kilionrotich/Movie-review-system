"""
Generates a synthetic MovieLens-like dataset for offline use / demos.

The generated data mimics the structure of the real MovieLens Small dataset
(ml-latest-small) so all downstream code works identically.

Usage
-----
    python generate_sample_data.py          # writes to ./data/ml-latest-small/
    python generate_sample_data.py --n-users 200 --n-movies 500
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Film corpus (title, genres, year)
# ---------------------------------------------------------------------------

MOVIES_CORPUS = [
    ("Toy Story", "Animation|Children|Comedy", 1995),
    ("Jumanji", "Adventure|Children|Fantasy", 1995),
    ("GoldenEye", "Action|Adventure|Thriller", 1995),
    ("Four Rooms", "Comedy|Thriller", 1995),
    ("Get Shorty", "Comedy|Crime|Thriller", 1995),
    ("Copycat", "Crime|Drama|Horror|Mystery|Thriller", 1995),
    ("Sudden Death", "Action", 1995),
    ("Se7en", "Crime|Mystery|Thriller", 1995),
    ("The Usual Suspects", "Crime|Mystery|Thriller", 1995),
    ("Braveheart", "Action|Drama|War", 1995),
    ("Apollo 13", "Adventure|Drama|IMAX", 1995),
    ("Heat", "Action|Crime|Thriller", 1995),
    ("Sabrina", "Comedy|Romance", 1995),
    ("Tom and Huck", "Adventure|Children", 1995),
    ("Sudden Death", "Action", 1995),
    ("Casino", "Crime|Drama", 1995),
    ("Fargo", "Comedy|Crime|Drama|Thriller", 1996),
    ("Pulp Fiction", "Comedy|Crime|Drama|Thriller", 1994),
    ("The Silence of the Lambs", "Crime|Horror|Thriller", 1991),
    ("Schindler's List", "Drama|War", 1993),
    ("The Shawshank Redemption", "Crime|Drama", 1994),
    ("Forrest Gump", "Comedy|Drama|Romance|War", 1994),
    ("The Lion King", "Adventure|Animation|Children|Drama|Musical|IMAX", 1994),
    ("Speed", "Action|Romance|Thriller", 1994),
    ("True Lies", "Action|Adventure|Comedy|Romance|Thriller", 1994),
    ("The Matrix", "Action|Sci-Fi|Thriller", 1999),
    ("Fight Club", "Action|Crime|Drama|Thriller", 1999),
    ("American Beauty", "Comedy|Crime|Drama", 1999),
    ("The Sixth Sense", "Drama|Horror|Mystery|Thriller", 1999),
    ("Being John Malkovich", "Comedy|Drama|Fantasy", 1999),
    ("Magnolia", "Drama", 1999),
    ("Eyes Wide Shut", "Drama|Mystery|Romance|Thriller", 1999),
    ("The Green Mile", "Crime|Drama|Fantasy|Mystery", 1999),
    ("Star Wars: Episode I", "Action|Adventure|Sci-Fi", 1999),
    ("American Pie", "Comedy|Romance", 1999),
    ("The Blair Witch Project", "Horror|Mystery|Thriller", 1999),
    ("Gladiator", "Action|Adventure|Drama", 2000),
    ("Cast Away", "Adventure|Drama|Romance", 2000),
    ("Traffic", "Crime|Drama|Thriller", 2000),
    ("Requiem for a Dream", "Drama", 2000),
    ("Snatch", "Comedy|Crime|Thriller", 2000),
    ("Almost Famous", "Drama|Romance", 2000),
    ("Erin Brockovich", "Biography|Drama", 2000),
    ("O Brother Where Art Thou?", "Adventure|Comedy|Crime", 2000),
    ("Memento", "Mystery|Thriller", 2000),
    ("A Beautiful Mind", "Biography|Drama|Thriller", 2001),
    ("Lord of the Rings: Fellowship", "Adventure|Fantasy", 2001),
    ("Harry Potter and the Sorcerer's Stone", "Adventure|Children|Fantasy", 2001),
    ("Shrek", "Adventure|Animation|Children|Comedy|Fantasy|Romance", 2001),
    ("Monsters Inc", "Adventure|Animation|Children|Comedy|Fantasy", 2001),
    ("Mulholland Drive", "Crime|Drama|Fantasy|Mystery|Thriller", 2001),
    ("The Royal Tenenbaums", "Comedy|Drama", 2001),
    ("Amelie", "Comedy|Romance", 2001),
    ("Spider-Man", "Action|Adventure|Sci-Fi|Thriller|IMAX", 2002),
    ("The Pianist", "Biography|Drama|War", 2002),
    ("Chicago", "Comedy|Crime|Drama|Musical|Romance", 2002),
    ("Catch Me If You Can", "Crime|Drama|Thriller", 2002),
    ("The Hours", "Drama|Romance", 2002),
    ("Adaptation", "Comedy|Drama", 2002),
    ("Lord of the Rings: Two Towers", "Adventure|Fantasy", 2002),
    ("Pirates of the Caribbean", "Action|Adventure|Comedy|Fantasy", 2003),
    ("Lost in Translation", "Comedy|Drama|Romance", 2003),
    ("Kill Bill Vol 1", "Action|Crime|Thriller", 2003),
    ("Lord of the Rings: Return of the King", "Action|Adventure|Drama|Fantasy", 2003),
    ("Master and Commander", "Action|Adventure|Drama|War", 2003),
    ("Mystic River", "Crime|Drama|Mystery|Thriller", 2003),
    ("Big Fish", "Adventure|Drama|Fantasy|Romance", 2003),
    ("Eternal Sunshine of the Spotless Mind", "Drama|Romance|Sci-Fi", 2004),
    ("The Aviator", "Biography|Drama|Romance", 2004),
    ("Sideways", "Comedy|Drama|Romance", 2004),
    ("Collateral", "Action|Crime|Drama|Thriller", 2004),
    ("Hotel Rwanda", "Drama|History|War", 2004),
    ("Ray", "Biography|Drama|Music", 2004),
    ("Million Dollar Baby", "Drama|Sport", 2004),
    ("Batman Begins", "Action|Crime|Thriller|IMAX", 2005),
    ("Brokeback Mountain", "Drama|Romance", 2005),
    ("Capote", "Biography|Crime|Drama", 2005),
    ("Munich", "Drama|History|Thriller", 2005),
    ("Good Night and Good Luck", "Biography|Drama|History|Thriller", 2005),
    ("Crash", "Crime|Drama|Thriller", 2005),
    ("The Departed", "Crime|Drama|Thriller", 2006),
    ("Little Miss Sunshine", "Adventure|Comedy|Drama", 2006),
    ("Children of Men", "Action|Adventure|Drama|Sci-Fi|Thriller", 2006),
    ("Pan's Labyrinth", "Drama|Fantasy|War", 2006),
    ("The Queen", "Biography|Drama|History", 2006),
    ("Letters from Iwo Jima", "Drama|History|War", 2006),
    ("No Country for Old Men", "Crime|Drama|Thriller|Western", 2007),
    ("There Will Be Blood", "Drama", 2007),
    ("Michael Clayton", "Crime|Drama|Mystery|Thriller", 2007),
    ("Juno", "Comedy|Drama|Romance", 2007),
    ("Into the Wild", "Adventure|Biography|Drama", 2007),
    ("Atonement", "Drama|Romance|War", 2007),
    ("The Dark Knight", "Action|Crime|Drama|IMAX", 2008),
    ("Slumdog Millionaire", "Crime|Drama|Romance", 2008),
    ("The Wrestler", "Drama|Sport", 2008),
    ("Frost/Nixon", "Biography|Drama|History", 2008),
    ("Milk", "Biography|Drama|History", 2008),
    ("Up", "Adventure|Animation|Children|Drama", 2009),
    ("The Hurt Locker", "Action|Drama|Thriller|War", 2009),
    ("Inglourious Basterds", "Adventure|Drama|War", 2009),
    ("Avatar", "Action|Adventure|Sci-Fi|IMAX", 2009),
    ("Up in the Air", "Drama|Romance", 2009),
    ("A Serious Man", "Comedy|Drama", 2009),
    ("Toy Story 3", "Adventure|Animation|Children|Comedy|Fantasy|IMAX", 2010),
    ("The Social Network", "Drama", 2010),
    ("Black Swan", "Drama|Mystery|Thriller", 2010),
    ("Inception", "Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX", 2010),
    ("The King's Speech", "Biography|Drama|History", 2010),
    ("True Grit", "Drama|Western", 2010),
    ("Winter's Bone", "Crime|Drama|Mystery|Thriller", 2010),
    ("The Artist", "Comedy|Drama|Romance", 2011),
    ("The Descendants", "Comedy|Drama|Romance", 2011),
    ("Drive", "Crime|Drama", 2011),
    ("Midnight in Paris", "Comedy|Fantasy|Romance", 2011),
    ("Tinker Tailor Soldier Spy", "Drama|Mystery|Thriller", 2011),
    ("War Horse", "Drama|War", 2011),
    ("Argo", "Biography|Drama|Thriller", 2012),
    ("Lincoln", "Biography|Drama|History|War", 2012),
    ("Django Unchained", "Drama|Western", 2012),
    ("Zero Dark Thirty", "Drama|History|Thriller|War", 2012),
    ("The Master", "Drama", 2012),
    ("Silver Linings Playbook", "Comedy|Drama|Romance", 2012),
    ("12 Years a Slave", "Biography|Drama|History", 2013),
    ("Gravity", "Drama|Sci-Fi|Thriller|IMAX", 2013),
    ("The Wolf of Wall Street", "Biography|Comedy|Crime|Drama", 2013),
    ("Her", "Drama|Romance|Sci-Fi", 2013),
    ("Nebraska", "Adventure|Comedy|Drama", 2013),
    ("American Hustle", "Comedy|Crime|Drama", 2013),
    ("Boyhood", "Drama", 2014),
    ("Birdman", "Comedy|Drama", 2014),
    ("Whiplash", "Drama|Music", 2014),
    ("The Imitation Game", "Biography|Drama|Thriller|War", 2014),
    ("The Grand Budapest Hotel", "Adventure|Comedy|Crime|Drama|Mystery|Romance", 2014),
    ("Selma", "Biography|Drama|History", 2014),
    ("Mad Max: Fury Road", "Action|Adventure|Sci-Fi|Thriller|IMAX", 2015),
    ("The Revenant", "Action|Adventure|Drama|Western", 2015),
    ("Spotlight", "Biography|Crime|Drama|Mystery|Thriller", 2015),
    ("The Martian", "Drama|Sci-Fi|IMAX", 2015),
    ("Room", "Drama|Thriller", 2015),
    ("Ex Machina", "Drama|Sci-Fi|Thriller", 2015),
    ("Moonlight", "Drama|Romance", 2016),
    ("La La Land", "Comedy|Drama|Music|Romance", 2016),
    ("Arrival", "Drama|Mystery|Sci-Fi", 2016),
    ("Hacksaw Ridge", "Biography|Drama|History|War", 2016),
    ("Hell or High Water", "Crime|Drama|Western", 2016),
    ("Manchester by the Sea", "Drama", 2016),
    ("Get Out", "Horror|Mystery|Thriller", 2017),
    ("Lady Bird", "Comedy|Drama|Romance", 2017),
    ("The Shape of Water", "Adventure|Drama|Fantasy|Romance|Sci-Fi", 2017),
    ("Dunkirk", "Action|Drama|History|Thriller|War|IMAX", 2017),
    ("Three Billboards Outside Ebbing Missouri", "Crime|Drama", 2017),
    ("Phantom Thread", "Drama|Romance", 2017),
    ("The Post", "Biography|Drama|History|Thriller", 2017),
    ("A Star Is Born", "Drama|Music|Romance", 2018),
    ("Green Book", "Biography|Comedy|Drama", 2018),
    ("Roma", "Drama", 2018),
    ("Black Panther", "Action|Adventure|Sci-Fi|IMAX", 2018),
    ("BlacKkKlansman", "Biography|Comedy|Crime|Drama|Thriller", 2018),
    ("Vice", "Biography|Comedy|Drama|History", 2018),
    ("Parasite", "Comedy|Crime|Drama|Thriller", 2019),
    ("Joker", "Crime|Drama|Thriller", 2019),
    ("1917", "Drama|War|IMAX", 2019),
    ("The Irishman", "Biography|Crime|Drama", 2019),
    ("Once Upon a Time in Hollywood", "Comedy|Drama", 2019),
    ("Marriage Story", "Drama|Romance", 2019),
    ("Nomadland", "Drama", 2020),
    ("Minari", "Drama", 2020),
    ("Promising Young Woman", "Crime|Drama|Mystery|Thriller", 2020),
    ("The Trial of the Chicago 7", "Biography|Drama|History|Thriller", 2020),
    ("Mank", "Biography|Drama|History", 2020),
    ("Sound of Metal", "Drama|Music", 2020),
    ("CODA", "Drama|Music|Romance", 2021),
    ("The Power of the Dog", "Drama|Western", 2021),
    ("Belfast", "Biography|Drama|Romance", 2021),
    ("Dune", "Action|Adventure|Drama|Sci-Fi|IMAX", 2021),
    ("Tick, Tick... Boom!", "Biography|Drama|Music|Romance", 2021),
    ("Drive My Car", "Drama|Romance", 2021),
    ("Everything Everywhere All at Once", "Action|Adventure|Comedy|Sci-Fi", 2022),
    ("The Banshees of Inisherin", "Comedy|Drama", 2022),
    ("Tár", "Drama|Music", 2022),
    ("All Quiet on the Western Front", "Drama|War", 2022),
    ("The Fabelmans", "Biography|Drama", 2022),
    ("Women Talking", "Drama|Mystery", 2022),
]

TAGS_CORPUS = [
    "classic", "critically acclaimed", "oscar winner", "cult classic",
    "slow burn", "visually stunning", "feel-good", "thought-provoking",
    "dark", "funny", "romantic", "action-packed", "suspenseful",
    "inspirational", "based on true story", "sci-fi", "fantasy",
    "historical", "based on book", "sequel", "family friendly",
    "independent", "foreign film", "animated", "musical",
    "crime", "heist", "dystopian", "coming of age", "war",
    "psychological thriller", "comedy", "biopic", "mystery",
]


def generate_dataset(
    n_users: int = 500,
    n_movies: int | None = None,
    n_ratings_per_user: tuple[int, int] = (20, 150),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic MovieLens-like datasets.

    Parameters
    ----------
    n_users : int
    n_movies : int or None – if None, use all movies in MOVIES_CORPUS
    n_ratings_per_user : tuple – (min, max) ratings per user
    seed : int

    Returns
    -------
    tuple
        (ratings, movies, tags)
    """
    rng = np.random.default_rng(seed)

    # ── Movies ────────────────────────────────────────────────────────────────
    corpus = MOVIES_CORPUS
    if n_movies is not None:
        corpus = corpus[:n_movies]

    movies_data = []
    for idx, (title, genres, year) in enumerate(corpus, start=1):
        movies_data.append({
            "movieId": idx,
            "title": f"{title} ({year})",
            "genres": genres,
        })
    movies = pd.DataFrame(movies_data)

    # ── Ratings ───────────────────────────────────────────────────────────────
    # Assign latent "quality" to movies so ratings cluster realistically
    movie_quality = rng.normal(3.5, 0.7, size=len(movies)).clip(0.5, 5.0)

    rating_rows = []
    base_ts = int(time.mktime(time.strptime("2010-01-01", "%Y-%m-%d")))
    end_ts = int(time.mktime(time.strptime("2023-12-31", "%Y-%m-%d")))

    for user_id in range(1, n_users + 1):
        n_rated = int(rng.integers(n_ratings_per_user[0], n_ratings_per_user[1]))
        rated_movie_idx = rng.choice(len(movies), size=min(n_rated, len(movies)), replace=False)

        # Each user has a bias (strict vs lenient rater)
        user_bias = rng.normal(0, 0.3)

        for midx in rated_movie_idx:
            raw = movie_quality[midx] + user_bias + rng.normal(0, 0.5)
            # Round to nearest 0.5
            rating = round(max(0.5, min(5.0, raw)) * 2) / 2
            ts = int(rng.integers(base_ts, end_ts))
            rating_rows.append({
                "userId": user_id,
                "movieId": int(movies.iloc[midx]["movieId"]),
                "rating": rating,
                "timestamp": ts,
            })

    ratings = pd.DataFrame(rating_rows)

    # ── Tags ──────────────────────────────────────────────────────────────────
    tag_rows = []
    n_tags = max(200, n_users // 2)
    for _ in range(n_tags):
        user_id = int(rng.integers(1, n_users + 1))
        movie_id = int(movies.sample(1, random_state=int(rng.integers(1e6)))["movieId"].iloc[0])
        tag = random.choice(TAGS_CORPUS)
        ts = int(rng.integers(base_ts, end_ts))
        tag_rows.append({
            "userId": user_id,
            "movieId": movie_id,
            "tag": tag,
            "timestamp": ts,
        })

    tags = pd.DataFrame(tag_rows).drop_duplicates(subset=["userId", "movieId", "tag"])

    return ratings, movies, tags


def save_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    output_dir: str,
) -> None:
    """Write CSVs in MovieLens format."""
    os.makedirs(output_dir, exist_ok=True)
    ratings.to_csv(os.path.join(output_dir, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(output_dir, "movies.csv"), index=False)
    tags.to_csv(os.path.join(output_dir, "tags.csv"), index=False)
    print(f"Saved {len(ratings):,} ratings, {len(movies):,} movies, {len(tags):,} tags → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MovieLens-like data.")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--n-movies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data", "ml-latest-small"))
    args = parser.parse_args()

    ratings, movies, tags = generate_dataset(
        n_users=args.n_users,
        n_movies=args.n_movies,
        seed=args.seed,
    )
    save_dataset(ratings, movies, tags, args.output_dir)
