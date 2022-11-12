import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class ReRanker:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/sentence-t5-large")

    def rerank(self, input_dict : dict):
        for question in input_dict.keys():
            summaries_articles = input_dict[question]                
            docs = [summary_article[0] for summary_article in summaries_articles]
            doc_embedding = self.model.encode(question)
            candidate_embeddings = self.model.encode(docs)
            distances = cosine_similarity([doc_embedding], candidate_embeddings)[0]
            distances = [float(s) for s in distances]
            # print(len(distances))

            top_results = list(zip(distances, summaries_articles[0], summaries_articles[1]))
            new_out = sorted(top_results, key=lambda x: x[0], reverse=True)
            input_dict[question] = new_out
        # print(len(input_dict.keys()))  
        return input_dict

