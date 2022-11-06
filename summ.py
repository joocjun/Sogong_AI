from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summ(article):
    tokenizer_kwargs = {'truncation':True}
    summarizer = pipeline("summarization",  model="t5-base", tokenizer="t5-base")
    ARTICLE = article
    summ = summarizer(ARTICLE, max_length=512, min_length=30, do_sample=False,**tokenizer_kwargs)[-1]["summary_text"]
    return summ 
