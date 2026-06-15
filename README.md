🎵 Mood-Based Music Recommender

A Machine Learning powered music recommendation system that suggests songs based on a user's current mood.

The application analyzes Spotify audio features, classifies songs into mood categories, groups similar tracks using K-Means Clustering, and generates personalized recommendations using Cosine Similarity.

📌Features
-> Mood-based song recommendations
-> Supports multiple moods:
Happy
Sad
Calm
Energetic
🤖 Machine Learning powered recommendations
📊 K-Means clustering visualization
🎯 Similarity-based song matching using Cosine Similarity
🌐 Interactive Flask web application
🎨 Modern responsive UI

🛠️ Technologies Used
Python
Flask
Pandas
NumPy
Scikit-Learn
Matplotlib
HTML
CSS

📂 Dataset
The project uses a Spotify songs dataset containing audio features such as:
Danceability
Energy
Valence
Tempo
These features are used to determine mood patterns and generate recommendations.

🧠 Machine Learning Workflow

1. Data Preprocessing
Missing value handling
Feature selection
Data cleaning
Feature scaling using StandardScaler

2. Mood Classification

Songs are assigned moods using Energy and Valence values:
| Mood      | Condition       |
| --------- | --------------- |
| Happy     | Valence > 0.7   |
| Sad       | Valence < 0.3   |
| Energetic | Energy > 0.7    |
| Calm      | Remaining songs |

3. K-Means Clustering

Songs are clustered based on:

Danceability
Energy
Valence
Tempo
K-Means is applied with:
n_clusters = 4

This groups songs with similar audio characteristics into distinct clusters.

4. Cosine Similarity
To generate recommendations:

A seed song is selected from the chosen mood category.
Cosine Similarity is calculated between songs.
Most similar songs are returned as recommendations.

This ensures recommendations are both mood-consistent and musically similar.

📊 Cluster Visualization

The scatter plot below shows how songs are grouped using K-Means Clustering based on Energy and Valence.

X-axis → Energy
Y-axis → Valence
Colors represent different clusters

The visualization helps understand how songs are distributed across mood-related audio features.

🚀 Application Flow

User Selects Mood
        │
        ▼
Mood Filtering
        │
        ▼
K-Means Cluster Analysis
        │
        ▼
Cosine Similarity Matching
        │
        ▼
Top Song Recommendations

🎨 User Interface
Home Page
Mood selection interface
Dataset statistics
Cluster visualization
Recommendation Page
Personalized playlist
Song rankings
Artist information
Similarity scores
Quick mood switching


📁 Project Structure

Mood-Music-Recommender/
│
├── app.py
├── spotify.csv
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
│
├── static/
│   └── cluster_plot.png
│
├── requirements.txt
└── README.md

⚙️ Installation
Clone Repository
git clone https://github.com/your-username/mood-music-recommender.git
cd mood-music-recommender
Install Dependencies
pip install -r requirements.txt
Run Application
python app.py

Open:

http://127.0.0.1:5000

📈 Future Enhancements
Spotify API Integration
User Login & Playlists
Deep Learning Recommendation Engine
Collaborative Filtering
Hybrid Recommendation System
Real-time Mood Detection
Emotion Detection using Webcam
Song Preview Support
