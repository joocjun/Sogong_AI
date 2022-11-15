from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
from utils import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json



class SpeedyPipeline():
    def __init__(self):
        # print_now()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5tokenizer = AutoTokenizer.from_pretrained("t5-base",truncation=True,torchscript=True)
        t5model = AutoModelWithLMHead.from_pretrained("t5-base",torchscript=True)
        self.summarizer = pipeline("summarization",  model=t5model, tokenizer=t5tokenizer,device=self.device)
        self.model = SentenceTransformer("sentence-transformers/sentence-t5-base")

    def process(self,input):
        sub_questions = []
        documents = []
        for subproblem in input["explanation"].values(): # input["explanation"]["0"]
            sub_questions.append(subproblem["sub_question"]) 
            for document in subproblem["evidence_document"].values(): # input["explanation"]["0"]["evidence_document"]
                documents.append(document["document"])
        summaries = self.summary_pipeline(documents,summarizer=self.summarizer)
        doc_embeddings = self.model.encode(sub_questions)
        candidate_embeddings = self.model.encode(summaries) 
        summaries = [summaries[i:i+5] for i in range(0, len(summaries), 5)]
        candidate_embeddings = [candidate_embeddings[i:i + 5] for i in range(0, len(candidate_embeddings), 5)]
        assert len(candidate_embeddings) == len(doc_embeddings), print(len(candidate_embeddings),len(doc_embeddings))
        idx = 0
        for de, ce, summ in zip(doc_embeddings,candidate_embeddings,summaries):
            distances = cosine_similarity([de], ce)[0]
            distances = [float(s) for s in distances]
            top_results = list(zip(distances, summ))
            new_out = sorted(top_results, key=lambda x: x[0], reverse=True)
            inner_idx = 0
            for dist,sum in new_out:
                input["explanation"][str(idx)]["evidence_document"][str(inner_idx)]["document"] = sum
                input["explanation"][str(idx)]["evidence_document"][str(inner_idx)]["score"] = dist
                inner_idx += 1
            idx+=1
        return input



    def summary_pipeline(self,articles, summarizer):
        summ = summarizer(articles, max_length=128, min_length=20, return_text=True)
        out = [sum["summary_text"] for sum in summ]
        return out

    def process_one(self,input):
        sub_questions = []
        documents = []
        for subproblem in input["explanation"].values(): # input["explanation"]["0"]
            sub_questions.append(subproblem["sub_question"]) 
            for document in subproblem["evidence_document"].values(): # input["explanation"]["0"]["evidence_document"]
                documents.append(self.summary_pipeline([document["document"]],summarizer=self.summarizer)[0])
        doc_embeddings = self.model.encode(sub_questions)
        candidate_embeddings = self.model.encode(documents) 
        summaries = [documents[i:i+5] for i in range(0, len(documents), 5)]
        candidate_embeddings = [candidate_embeddings[i:i + 5] for i in range(0, len(candidate_embeddings), 5)]
        assert len(candidate_embeddings) == len(doc_embeddings), print(len(candidate_embeddings),len(doc_embeddings))
        idx = 0
        for de, ce, summ in zip(doc_embeddings,candidate_embeddings,summaries):
            distances = cosine_similarity([de], ce)[0]
            distances = [float(s) for s in distances]
            top_results = list(zip(distances, summ))
            new_out = sorted(top_results, key=lambda x: x[0], reverse=True)
            inner_idx = 0
            for dist,sum in new_out:
                input["explanation"][str(idx)]["evidence_document"][str(inner_idx)]["document"] = sum
                input["explanation"][str(idx)]["evidence_document"][str(inner_idx)]["score"] = dist
                inner_idx += 1
            idx+=1
        return input