import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class ReRanker:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/sentence-t5-large")

    def rerank(self, question, explanation, docs):
        doc_embedding = self.model.encode(question+"\n"+explanation)
        candidate_embeddings = self.model.encode(docs)
        distances = cosine_similarity([doc_embedding], candidate_embeddings)[0]

        top_results = zip(docs, distances)
        return sorted(top_results, key=lambda x: x[1], reverse=True)

