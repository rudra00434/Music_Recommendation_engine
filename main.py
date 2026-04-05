from preprocessing import load_data, preprocess
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender

import numpy as np

# Load + preprocess
df = preprocess(load_data("songs.csv"))

# Feature matrix (example)
features = df[['tempo', 'energy', 'danceability']].values

# Content model
content = ContentRecommender()
content.fit(features)

# Fake user-item matrix (replace with real)
user_item = np.random.rand(100, len(df))

collab = CollaborativeRecommender()
collab.fit(user_item)

# Hybrid system
hybrid = HybridRecommender(content, collab)

# Recommend
print(hybrid.recommend(user_id=1, song_index=10))
