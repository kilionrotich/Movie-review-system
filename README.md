# 🎬 Movie Recommendation System

A full-stack Movie Recommendation System built on the **MovieLens** dataset.  
It combines **Collaborative Filtering** (SVD matrix factorisation) and **Content-Based Filtering** (TF-IDF) with a **Streamlit** web app that lets users explore movies and get personalised recommendations.

![App Screenshot](https://github.com/user-attachments/assets/81bde843-b5b1-43a2-bf64-09a6ff3c4c97)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📥 **Data Loading** | Auto-downloads MovieLens Small dataset; falls back to synthetic data offline |
| 🧹 **Data Cleaning** | Deduplication, invalid-rating removal, title/genre parsing |
| 📊 **EDA Dashboard** | Rating distributions, genre popularity, activity over time, top-rated movies |
| 🤝 **Collaborative Filtering** | SVD matrix factorisation via `scikit-surprise` |
| 🎭 **Content-Based Filtering** | TF-IDF on genres + user tags, cosine similarity |
| 🔀 **Hybrid Mode** | Weighted blend of CF and CBF scores |
| 📏 **Evaluation** | RMSE, MAE, Precision@K, Recall@K |
| 🌐 **Streamlit App** | Interactive UI with 6 recommendation modes |

---

## 🗂️ Project Structure

```
Movie-review-system/
├── app.py                    # Streamlit web application
├── train.py                  # One-shot training script
├── generate_sample_data.py   # Synthetic MovieLens-like dataset generator
├── requirements.txt
├── src/
│   ├── data_loader.py        # Download / load MovieLens data
│   ├── preprocessing.py      # Data cleaning & feature engineering
│   ├── eda.py                # Exploratory data analysis plots
│   ├── collaborative_filtering.py  # SVD model (scikit-surprise)
│   ├── content_based.py      # TF-IDF content-based model (scikit-learn)
│   ├── evaluation.py         # RMSE, MAE, Precision@K, Recall@K
│   └── recommender.py        # High-level recommender (CF + CBF + hybrid)
├── tests/
│   └── test_recommender.py   # Unit tests (no internet required)
└── data/                     # Auto-generated: dataset, model artifacts, EDA plots
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `scikit-surprise` requires `numpy < 2.0`.  
> The `requirements.txt` already pins `numpy>=1.23,<2.0`.

### 2. Train the models

```bash
python train.py
```

This will:
- Download the MovieLens Small dataset (or generate synthetic data if offline)
- Run EDA and save plots to `data/eda_plots/`
- Train the SVD and TF-IDF models
- Print evaluation metrics (RMSE, MAE, Precision@K, Recall@K)
- Save model artifacts to `data/`

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Models & Evaluation

### Collaborative Filtering (SVD)

Uses Singular Value Decomposition to decompose the user–item rating matrix into latent factors.  
Implemented with [`scikit-surprise`](https://surpriselib.com/).

**Hyperparameters:**
- `n_factors=100` — number of latent dimensions
- `n_epochs=20` — SGD training epochs  
- `lr_all=0.005`, `reg_all=0.02` — learning rate and regularisation

**Metrics on 20% held-out test set (synthetic data):**

| Metric | Value |
|---|---|
| RMSE | ~0.52 |
| MAE | ~0.41 |
| Precision@10 | ~0.70 |
| Recall@10 | ~0.79 |

### Content-Based Filtering (TF-IDF)

Combines movie genres and user-generated tags into a "soup" string, then applies TF-IDF vectorisation with bigrams.  
Recommendations are ranked by cosine similarity of feature vectors.

---

## 🌐 Streamlit App Modes

| Mode | Description |
|---|---|
| 🔍 **Search & Browse** | Full-text title search + genre filter |
| 👤 **Recommend for User (CF)** | Personalised recommendations using SVD |
| 🎭 **Similar Movies (CBF)** | Find movies similar to a title via TF-IDF |
| ❤️ **Based on Liked Movies (CBF)** | Recommendations from a list of liked titles |
| 🔀 **Hybrid Recommendations** | Weighted blend of CF + CBF |
| 📊 **EDA Dashboard** | Interactive exploratory data analysis |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | TF-IDF, cosine similarity |
| `scikit-surprise` | SVD collaborative filtering |
| `streamlit` | Web application |
| `matplotlib`, `seaborn` | Visualisation |
| `requests` | Dataset download |
| `pytest` | Testing |

---

## 📁 Data

The system uses the **[MovieLens Small](https://grouplens.org/datasets/movielens/latest/)** dataset (~100K ratings, 9K movies, 600 users).  
If the download fails (e.g., in an offline environment), a synthetic dataset with the same schema is generated automatically via `generate_sample_data.py`.
