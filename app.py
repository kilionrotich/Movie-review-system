"""
Streamlit app for the Movie Recommendation System.

Run with:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.recommender import MovieRecommender

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Load / train model (cached)
# ──────────────────────────────────────────────────────────────────────────────
STATE_PATH = os.path.join(os.path.dirname(__file__), "data", "recommender_state.pkl")
CF_PATH = os.path.join(os.path.dirname(__file__), "data", "cf_model.pkl")


@st.cache_resource(show_spinner="Loading recommendation models…")
def get_recommender() -> MovieRecommender:
    rec = MovieRecommender()
    if os.path.exists(STATE_PATH) and os.path.exists(CF_PATH):
        rec.load(STATE_PATH)
    else:
        st.info("First run: downloading data and training models. This may take a few minutes…")
        rec.fit()
        rec.save(STATE_PATH)
    return rec


rec = get_recommender()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Movie Recommender")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Choose recommendation mode",
    [
        "Search & Browse",
        "Recommend for User (CF)",
        "Similar Movies (CBF)",
        "Based on Liked Movies (CBF)",
        "Hybrid Recommendations",
        "EDA Dashboard",
    ],
)
st.sidebar.markdown("---")
n_recs = st.sidebar.slider("Number of recommendations", 5, 30, 10, step=5)

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def render_recommendations(df: pd.DataFrame, score_col: str = "score") -> None:
    """Render a recommendations DataFrame as a styled table with a bar chart."""
    if df.empty:
        st.warning("No recommendations found. Try adjusting your input.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        if score_col in df.columns:
            fig, ax = plt.subplots(figsize=(5, max(3, len(df) * 0.35)))
            title_col = "title" if "title" in df.columns else df.columns[1]
            titles = df[title_col].str[:30].tolist()
            scores = df[score_col].tolist()
            ax.barh(titles[::-1], scores[::-1], color="steelblue")
            ax.set_xlabel(score_col.replace("_", " ").title())
            ax.set_title("Recommendation Scores")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────────

st.title("Movie Recommendation System")
st.caption("Powered by SVD Collaborative Filtering + TF-IDF Content-Based Filtering on MovieLens data.")

# ── Search & Browse ──────────────────────────────────────────────────────────
if mode == "Search & Browse":
    st.header("Search & Browse Movies")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search by movie title", placeholder="e.g. Matrix, Star Wars, Toy Story")
    with col2:
        genre_filter = st.selectbox("Filter by genre", ["All"] + rec.list_genres())

    if query:
        results = rec.search_movies(query, top_n=50)
        if genre_filter != "All":
            mask = results["movieId"].isin(
                rec.movies[rec.movies["genres_list"].apply(lambda gl: genre_filter in gl)]["movieId"]
            )
            results = results[mask]
        st.write(f"Found **{len(results)}** matching movies:")
        st.dataframe(results.head(30), use_container_width=True, hide_index=True)
    elif genre_filter != "All":
        results = rec.movies_by_genre(genre_filter, top_n=30)
        st.write(f"Top movies in **{genre_filter}**:")
        st.dataframe(results, use_container_width=True, hide_index=True)
    else:
        st.info("Enter a movie title or pick a genre to browse.")

# ── Recommend for User (CF) ──────────────────────────────────────────────────
elif mode == "Recommend for User (CF)":
    st.header("Collaborative Filtering – Personalised Recommendations")
    st.markdown(
        "Enter a **User ID** to get personalised recommendations based on ratings "
        "from similar users."
    )

    user_ids = sorted(rec.ratings["userId"].unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.number_input(
            "User ID",
            min_value=int(min(user_ids)),
            max_value=int(max(user_ids)),
            value=int(user_ids[0]),
            step=1,
        )
    with col2:
        st.metric("Total users", f"{len(user_ids):,}")

    if st.button("Get Recommendations", key="cf_btn", type="primary"):
        with st.spinner("Computing…"):
            recs = rec.recommend_for_user(int(user_id), n=n_recs)
        st.subheader(f"Top {n_recs} recommendations for User {user_id}")
        render_recommendations(recs, score_col="estimated_rating")

        # Show user's top-rated movies for context
        user_ratings = rec.ratings[rec.ratings["userId"] == user_id].copy()
        user_ratings = user_ratings.merge(
            rec.movies[["movieId", "clean_title" if "clean_title" in rec.movies.columns else "title"]],
            on="movieId",
        )
        title_col = "clean_title" if "clean_title" in user_ratings.columns else "title"
        top_user = user_ratings.nlargest(5, "rating")[[title_col, "rating"]]
        with st.expander("User's highest-rated movies (context)"):
            st.dataframe(top_user.rename(columns={title_col: "title"}), hide_index=True)

# ── Similar Movies (CBF) ─────────────────────────────────────────────────────
elif mode == "Similar Movies (CBF)":
    st.header("Content-Based Filtering – Similar Movies")
    st.markdown("Find movies similar to a title you enjoy.")

    title_input = st.text_input("Enter a movie title", placeholder="e.g. Toy Story")

    if title_input:
        movie_id = rec._resolve_movie_id(title_input)
        if movie_id is None:
            st.warning(f"No movie found matching '{title_input}'. Try the Search tab first.")
        else:
            title_col = "clean_title" if "clean_title" in rec.movies.columns else "title"
            matched = rec.movies[rec.movies["movieId"] == movie_id]
            st.success(
                f"Found: **{matched.iloc[0][title_col]}** "
                f"({matched.iloc[0].get('year', 'N/A')}) – {matched.iloc[0]['genres']}"
            )
            with st.spinner("Finding similar movies…"):
                similar = rec.similar_to(movie_id, n=n_recs)
            st.subheader(f"Movies similar to '{matched.iloc[0][title_col]}'")
            render_recommendations(similar, score_col="similarity_score")

# ── Based on Liked Movies ─────────────────────────────────────────────────────
elif mode == "Based on Liked Movies (CBF)":
    st.header("Content-Based Filtering – Based on Your Liked Movies")
    st.markdown(
        "Enter one or more movie titles you enjoy (comma-separated) and "
        "we'll find movies with similar genres and themes."
    )

    liked_input = st.text_area(
        "Movies you like (one per line or comma-separated)",
        placeholder="Toy Story\nThe Matrix\nFight Club",
        height=120,
    )

    if st.button("Find Recommendations", key="cbf_btn", type="primary") and liked_input:
        titles = [t.strip() for t in liked_input.replace(",", "\n").split("\n") if t.strip()]
        resolved = []
        unresolved = []
        for t in titles:
            mid = rec._resolve_movie_id(t)
            if mid:
                resolved.append(mid)
            else:
                unresolved.append(t)

        if unresolved:
            st.warning(f"Could not find: {', '.join(unresolved)}")
        if not resolved:
            st.error("None of the provided titles were found in the dataset.")
        else:
            with st.spinner("Computing recommendations…"):
                cbf_recs = rec.recommend_from_liked(resolved, n=n_recs)
            st.subheader(f"Top {n_recs} recommendations based on your liked movies")
            render_recommendations(cbf_recs, score_col="similarity_score")

# ── Hybrid ────────────────────────────────────────────────────────────────────
elif mode == "Hybrid Recommendations":
    st.header("Hybrid Recommendations (CF + CBF)")
    st.markdown(
        "Blend collaborative filtering (user behaviour) and content-based filtering "
        "(genres & tags) for more robust recommendations."
    )

    col1, col2 = st.columns(2)
    with col1:
        user_ids = sorted(rec.ratings["userId"].unique().tolist())
        user_id_hybrid = st.number_input(
            "User ID (optional – leave 0 to skip)",
            min_value=0,
            max_value=int(max(user_ids)),
            value=0,
            step=1,
        )
    with col2:
        cf_weight = st.slider("CF weight (vs CBF weight)", 0.0, 1.0, 0.5, step=0.1)

    liked_input_hybrid = st.text_area(
        "Movies you like (optional, one per line)",
        placeholder="Toy Story\nStar Wars",
        height=80,
    )

    if st.button("Get Hybrid Recommendations", key="hybrid_btn", type="primary"):
        liked_ids = []
        if liked_input_hybrid.strip():
            titles = [t.strip() for t in liked_input_hybrid.replace(",", "\n").split("\n") if t.strip()]
            liked_ids = [rec._resolve_movie_id(t) for t in titles]
            liked_ids = [m for m in liked_ids if m]

        uid = int(user_id_hybrid) if user_id_hybrid > 0 else None

        if uid is None and not liked_ids:
            st.error("Please provide at least a User ID or liked movies.")
        else:
            with st.spinner("Computing hybrid recommendations…"):
                hybrid_recs = rec.hybrid_recommend(
                    user_id=uid,
                    liked_titles_or_ids=liked_ids if liked_ids else None,
                    n=n_recs,
                    cf_weight=cf_weight,
                )
            st.subheader(f"Top {n_recs} hybrid recommendations")
            render_recommendations(hybrid_recs, score_col="score")

# ── EDA Dashboard ─────────────────────────────────────────────────────────────
elif mode == "EDA Dashboard":
    st.header("Exploratory Data Analysis Dashboard")

    ratings = rec.ratings
    movies = rec.movies

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings", f"{len(ratings):,}")
    col2.metric("Unique Users", f"{ratings['userId'].nunique():,}")
    col3.metric("Unique Movies", f"{ratings['movieId'].nunique():,}")
    col4.metric("Mean Rating", f"{ratings['rating'].mean():.2f}")

    st.markdown("---")

    eda_dir = os.path.join(os.path.dirname(__file__), "data", "eda_plots")
    plot_files = {
        "Rating Distribution": "rating_distribution.png",
        "Ratings per User": "ratings_per_user.png",
        "Ratings per Movie": "ratings_per_movie.png",
        "Ratings Over Time": "ratings_over_time.png",
        "Top Rated Movies": "top_rated_movies.png",
        "Genre Popularity": "genre_popularity.png",
    }

    # If EDA plots exist, load them; otherwise generate on-the-fly
    from src.eda import run_all_eda

    existing = {name: os.path.join(eda_dir, fname)
                for name, fname in plot_files.items()
                if os.path.exists(os.path.join(eda_dir, fname))}

    if not existing:
        with st.spinner("Generating EDA plots…"):
            figs = run_all_eda(ratings, movies, save=True)
        st.success("EDA plots generated!")
        existing = {name: os.path.join(eda_dir, fname)
                    for name, fname in plot_files.items()
                    if os.path.exists(os.path.join(eda_dir, fname))}

    tab_names = list(existing.keys())
    if tab_names:
        tabs = st.tabs(tab_names)
        for tab, (name, path) in zip(tabs, existing.items()):
            with tab:
                st.image(path, use_container_width=True)

    # Genre distribution pie chart (live)
    if "genres_list" in movies.columns:
        st.subheader("Genre Distribution")
        genre_counts: dict = {}
        for gl in movies["genres_list"]:
            for g in gl:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_s = pd.Series(genre_counts).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        top_genres = genre_s.head(10)
        other = genre_s.iloc[10:].sum()
        values = list(top_genres.values) + [other]
        labels = list(top_genres.index) + ["Other"]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.set_title("Top Genre Distribution")
        st.pyplot(fig)
        plt.close(fig)

    # Sparsity info
    sparsity = 1 - len(ratings) / (ratings["userId"].nunique() * ratings["movieId"].nunique())
    st.info(f"**Matrix Sparsity**: {sparsity:.2%} — {len(ratings):,} ratings out of "
            f"{ratings['userId'].nunique() * ratings['movieId'].nunique():,} possible entries.")
