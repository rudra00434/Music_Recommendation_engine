class HybridRecommender:
    def __init__(self, content_model, collab_model, alpha=0.6):
        self.content_model = content_model
        self.collab_model = collab_model
        self.alpha = alpha

    def recommend(self, user_id, song_index, top_n=5):
        content_rec = self.content_model.recommend(song_index, top_n*2)
        collab_rec = self.collab_model.recommend(user_id, top_n*2)

        # Merge scores
        scores = {}

        for i, rank in enumerate(content_rec):
            scores[rank] = scores.get(rank, 0) + self.alpha * (1/(i+1))

        for i, rank in enumerate(collab_rec):
            scores[rank] = scores.get(rank, 0) + (1-self.alpha) * (1/(i+1))

        # Sort final
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [i[0] for i in ranked[:top_n]]
