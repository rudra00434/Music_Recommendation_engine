from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self):
        self.similarity_matrix = None

    def fit(self, feature_matrix):
        self.similarity_matrix = cosine_similarity(feature_matrix)

    def recommend(self, song_index, top_n=5):
        scores = list(enumerate(self.similarity_matrix[song_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [i[0] for i in scores[1:top_n+1]]
