from scipy.sparse.linalg import svds
import numpy as np

class CollaborativeRecommender:
    def __init__(self, k=50):
        self.k = k

    def fit(self, user_item_matrix):
        U, sigma, Vt = svds(user_item_matrix, k=self.k)
        self.pred_matrix = np.dot(np.dot(U, np.diag(sigma)), Vt)

    def recommend(self, user_id, top_n=5):
        user_ratings = self.pred_matrix[user_id]
        return np.argsort(-user_ratings)[:top_n]
