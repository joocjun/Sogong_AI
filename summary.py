from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import *

def summary_pipeline(article):
    tokenizer_kwargs = {'truncation':True}
    # summarizer = pipeline("summarization",  model="t5-base", tokenizer="t5-base")
    # summarizer = pipeline("summarization",  model="t5-base", tokenizer="t5-base")
    summarizer = pipeline("summarization",  model="google/pegasus-large", tokenizer="google/pegasus-large")
    ARTICLE = article
    summ = summarizer(ARTICLE, max_length=512, min_length=30, do_sample=False,**tokenizer_kwargs)[-1]["summary_text"]
    return summ 

class Summarizer():
    def __init__(self):
        print_now()

    def summarize(self, article):
        summary = summary_pipeline(article)
        return [summary,article]
        