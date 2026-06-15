"""
Mood-Based Music Recommendation System
Flask Web Application
"""

import os
import random
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Flask App Initialization
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spotify.csv")
PLOT_PATH = os.path.join(BASE_DIR, "static", "cluster_plot.png")

os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)


# Custom Jinja2 filter
@app.template_filter("format_number")
def format_number(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value


def load_and_preprocess():
    """Load dataset, clean it, engineer features, cluster, and build similarity matrix."""

    # ── Load ──────────────────────────────────
    df = pd.read_csv(DATA_PATH)

    # Rename columns to match expected names
    df.rename(columns={"artists": "artist_name"}, inplace=True)

    # Keep only required columns
    required = ["track_name", "artist_name", "danceability", "energy", "valence", "tempo"]
    df = df[required].copy()

    # ── Handle Missing Values ─────────────────
    df.dropna(subset=["track_name", "artist_name"], inplace=True)
    df["danceability"].fillna(df["danceability"].median(), inplace=True)
    df["energy"].fillna(df["energy"].median(), inplace=True)
    df["valence"].fillna(df["valence"].median(), inplace=True)
    df["tempo"].fillna(df["tempo"].median(), inplace=True)

    df.reset_index(drop=True, inplace=True)

    # ─────────────────────────────────────────
    # 2. FEATURE ENGINEERING — Mood Labels
    # ─────────────────────────────────────────
    def assign_mood(row):
        v = row["valence"]
        e = row["energy"]
        if v > 0.7:
            return "Happy"
        elif v < 0.3:
            return "Sad"
        elif e > 0.7:
            return "Energetic"
        else:
            return "Calm"

    df["mood"] = df.apply(assign_mood, axis=1)

    # ─────────────────────────────────────────
    # 3A. SCALING
    # ─────────────────────────────────────────
    features = ["danceability", "energy", "valence", "tempo"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # ─────────────────────────────────────────
    # 3B. K-MEANS CLUSTERING
    # ─────────────────────────────────────────
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_features)

    # ─────────────────────────────────────────
    # 3C. COSINE SIMILARITY MATRIX
    #     (computed on a sample to keep memory manageable)
    # ─────────────────────────────────────────
    sample_size = min(5000, len(df))
    sample_idx = df.sample(n=sample_size, random_state=42).index
    sample_df = df.loc[sample_idx].copy().reset_index(drop=True)
    sample_scaled = scaler.transform(sample_df[features])
    similarity_matrix = cosine_similarity(sample_scaled)

    # ─────────────────────────────────────────
    # 4. VISUALIZATION — Cluster Scatter Plot
    # ─────────────────────────────────────────
    generate_cluster_plot(df)

    return df, sample_df, similarity_matrix


def generate_cluster_plot(df):
    """Generate and save the energy vs valence cluster scatter plot."""
    cluster_colors = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1", 3: "#96CEB4"}
    cluster_labels = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3"}

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        ax.scatter(
            subset["energy"],
            subset["valence"],
            c=cluster_colors.get(cluster_id, "#ffffff"),
            label=cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
            alpha=0.45,
            s=12,
            edgecolors="none",
        )

    ax.set_xlabel("Energy →", color="white", fontsize=12, labelpad=8)
    ax.set_ylabel("Valence →", color="white", fontsize=12, labelpad=8)
    ax.set_title("Song Clusters — Energy vs Valence", color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    legend = ax.legend(
        facecolor="#0f3460", edgecolor="#444", labelcolor="white", fontsize=10
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def recommend_songs(mood: str, df: pd.DataFrame, sample_df: pd.DataFrame,
                    similarity_matrix: np.ndarray, top_n: int = 5):
    """
    Filter songs by mood, then return top_n recommendations.
    Uses cosine similarity when enough songs are available in sample,
    otherwise falls back to random sampling.
    """
    mood_df = df[df["mood"] == mood]
    total_found = len(mood_df)

    if total_found == 0:
        return [], 0

    # Try similarity-based recommendation from sample set
    sample_mood = sample_df[sample_df["mood"] == mood]

    if len(sample_mood) >= top_n:
        # Pick a random seed song from the mood pool
        seed_idx = random.choice(sample_mood.index.tolist())
        sim_scores = list(enumerate(similarity_matrix[seed_idx]))
        # Sort by similarity (desc), skip seed itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != seed_idx]

        candidates = []
        for idx, score in sim_scores:
            if sample_df.iloc[idx]["mood"] == mood:
                candidates.append(idx)
            if len(candidates) >= top_n * 3:
                break

        if len(candidates) >= top_n:
            chosen = random.sample(candidates, top_n)
            recommendations = sample_df.iloc[chosen][["track_name", "artist_name", "mood", "cluster"]].copy()
            recommendations["similarity_score"] = [round(similarity_matrix[seed_idx][i], 3) for i in chosen]
            return recommendations.to_dict("records"), total_found

    # Fallback: random sample from full mood pool
    sample = mood_df.sample(n=min(top_n, total_found), random_state=random.randint(0, 9999))
    sample = sample[["track_name", "artist_name", "mood", "cluster"]].copy()
    sample["similarity_score"] = "N/A"
    return sample.to_dict("records"), total_found


# ─────────────────────────────────────────────
# BOOTSTRAP DATA ON STARTUP
# ─────────────────────────────────────────────
print("⏳  Loading and preprocessing dataset...")
try:
    DF, SAMPLE_DF, SIM_MATRIX = load_and_preprocess()
    MOOD_COUNTS = DF["mood"].value_counts().to_dict()
    print(f"✅  Dataset loaded — {len(DF):,} songs across {DF['mood'].nunique()} moods.")
    print(f"    Mood distribution: {MOOD_COUNTS}")
    DATA_LOADED = True
except Exception as exc:
    print(f"❌  Failed to load data: {exc}")
    DF, SAMPLE_DF, SIM_MATRIX = None, None, None
    MOOD_COUNTS = {}
    DATA_LOADED = False


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
MOODS = ["Happy", "Sad", "Calm", "Energetic"]
MOOD_EMOJI = {"Happy": "😊", "Sad": "😢", "Calm": "😌", "Energetic": "⚡"}
MOOD_COLORS = {
    "Happy":     "#f9ca24",
    "Sad":       "#6c5ce7",
    "Calm":      "#00b894",
    "Energetic": "#e17055",
}


@app.route("/")
def index():
    """Home page — mood selector."""
    if not DATA_LOADED:
        return render_template(
            "error.html",
            message="Dataset could not be loaded. Please check spotify.csv and restart."
        )
    return render_template(
        "index.html",
        moods=MOODS,
        mood_emoji=MOOD_EMOJI,
        mood_counts=MOOD_COUNTS,
        total_songs=len(DF),
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    """Recommendation result page."""
    if not DATA_LOADED:
        return render_template(
            "error.html",
            message="Dataset is not available. Please restart the server."
        )

    selected_mood = request.form.get("mood", "").strip()

    # Validate mood
    if selected_mood not in MOODS:
        return render_template(
            "result.html",
            error=f"Invalid mood '{selected_mood}'. Please go back and choose a valid option.",
            mood=selected_mood,
            mood_emoji=MOOD_EMOJI,
            mood_color=MOOD_COLORS.get(selected_mood, "#888"),
            recommendations=[],
            total_found=0,
            moods=MOODS,
        )

    recommendations, total_found = recommend_songs(
        selected_mood, DF, SAMPLE_DF, SIM_MATRIX, top_n=5
    )

    error = None
    if total_found == 0:
        error = f"No songs found in the '{selected_mood}' category. Try a different mood!"
    elif len(recommendations) == 0:
        error = "Something went wrong while fetching recommendations. Please try again."

    return render_template(
        "result.html",
        mood=selected_mood,
        mood_emoji=MOOD_EMOJI,
        mood_color=MOOD_COLORS.get(selected_mood, "#888"),
        recommendations=recommendations,
        total_found=total_found,
        mood_counts=MOOD_COUNTS,
        error=error,
        moods=MOODS,
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
