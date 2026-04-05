import faiss
import numpy as np

class FAISSRecommender:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def fit(self, vectors):
        self.index.add(np.array(vectors).astype('float32'))

    def recommend(self, query_vector, top_n=5):
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), top_n
        )
        return indices[0]
