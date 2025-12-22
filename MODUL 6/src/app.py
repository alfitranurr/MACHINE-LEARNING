import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches

st.set_page_config(page_title="Content-Based Movie Recommendation", layout="wide")

# ===============================
# Helpers
# ===============================

def remove_year_from_title(title: str) -> str:
    return re.sub(r"\(\d{4}\)", "", str(title)).strip()

def to_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

# Load data & load model
# ===============================

@st.cache_resource(show_spinner=True)
def load_data_and_model():
    # load data
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    df = pd.merge(movies, ratings, on="movieId")
    # buang kolom tidak dipakai, TAPI simpan movieId
    df = df.drop(columns=["timestamp", "userId", "rating"], errors="ignore")

    # Satu baris per film biar konsisten
    df.drop_duplicates(subset=["movieId"], inplace=True)

    # Preprocessing sama seperti training
    df["title"] = df["title"].apply(remove_year_from_title)
    df["genre"] = df["genres"].apply(lambda x: str(x).split("|"))
    df = df.drop(columns=["genres"], errors="ignore")
    df["title_norm"] = df["title"].apply(to_norm)

    tfidf_title = TfidfVectorizer = joblib.load("tfidf_title.pkl")
    tfidf_genre = TfidfVectorizer = joblib.load("tfidf_genre.pkl")
    nn = NearestNeighbors = joblib.load("nn_model.pkl")
    idx_map = pd.read_csv("titles_index.csv")  # cols: training_index, movieId, title
    idx_map = idx_map.drop_duplicates(subset=["movieId"]).sort_values("training_index")

    # re-order df agar PERSIS seperti saat training
    df = idx_map[["movieId"]].merge(df, on="movieId", how="left", sort=False)

    # X untuk title + genre
    X_title = tfidf_title.transform(df["title"])
    X_genre = tfidf_genre.transform(df["genre"].apply(lambda xs: " ".join(xs)))
    X = hstack([X_title, X_genre])

    # sanity check
    try:
        n_fit = getattr(nn, "_fit_X", None).shape[0]
        if n_fit != X.shape[0]:
            st.error(
                f"Model/data mismatch: model rows={n_fit}, current rows={X.shape[0]}. "
                "Pastikan titles_index.csv dibuat dari df yang sama saat fit."
            )
    except Exception:
        pass

    # mapping judul_norm -> index
    norm_to_index = {t: i for i, t in enumerate(df["title_norm"].values)}

    return df, X, nn, norm_to_index

df, X, nn, norm_to_index = load_data_and_model()

def recommend_movies(movie_title: str, num_recommendations: int = 5):
    query_norm = to_norm(movie_title)

    if query_norm not in norm_to_index:
        # saran judul mirip
        candidates = list(norm_to_index.keys())
        suggestions = get_close_matches(query_norm, candidates, n=5, cutoff=0.6)
        return [], (
            "Film tidak ditemukan. Mungkin maksud Anda: "
            + ", ".join([df["title"].iloc[norm_to_index[s]] for s in suggestions])
            if suggestions else "Film tidak ditemukan dalam dataset."
        )

    idx = norm_to_index[query_norm]
    k = min(num_recommendations + 50, X.shape[0])
    distances, indices = nn.kneighbors(X[idx], n_neighbors=k)

    recs, seen_idx, seen_titles = [], {idx}, {df["title"].iloc[idx]}
    for dist, i in zip(distances[0], indices[0]):
        title_i = df["title"].iloc[i]
        # dedup: skip diri sendiri & judul yang sama
        if i in seen_idx or title_i in seen_titles:
            continue

        similarity = 1.0 - float(dist)  # cosine similarity
        recs.append((title_i, similarity))
        seen_idx.add(i)
        seen_titles.add(title_i)

        if len(recs) >= num_recommendations:
            break

    if not recs:
        return [], "Tidak ada rekomendasi yang ditemukan."
    return recs, ""

st.title("Content-Based Movie Recommendation")

st.subheader(f"Daftar Film Tersedia ({df['title'].nunique()})")
with st.expander("Lihat semua judul"):
    st.caption("Daftar ini hanya untuk referensi – ketik judul di kotak pencarian di bawah.")
    st.dataframe(
        pd.DataFrame({"Title": sorted(df["title"].unique())}),
        use_container_width=True,
        hide_index=True
    )

movie_input = st.text_input("Judul film:")
num_recommendations = st.number_input("Jumlah rekomendasi:", min_value=1, max_value=20, value=5, step=1)

if movie_input:
    recs, msg = recommend_movies(movie_input, num_recommendations)
    if msg and "Film tidak ditemukan" in msg:
        st.warning(msg)
    elif recs:
        st.subheader(f"Rekomendasi untuk '{remove_year_from_title(movie_input)}':")
        for i, (title, sim) in enumerate(recs, start=1):
            st.write(f"{i}. {title} – Similarity: {sim:.3f}")
    if msg and "Tidak ada rekomendasi" in msg:
        st.info(msg)

