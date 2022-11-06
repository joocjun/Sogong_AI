# Sogong_AI
## 1. For GPT subquestion generation
### 1.1. gpt_request.py
gpt_request.py is a python script that generates subquestions using GPT-3 model.
use class Decoder's decode meethod to obtain subquestions.
- input: user_question (string), user_api_key **(string)**
- output: [question #1, evidence #1 , inference #1, question #2, evidence #2, ... , Answer] **(list of strings)**


## 2. For Generating Summary of Top 5 Web Document of a Question
### 2.1. summary.py
summary.py is a python script that generates summary of top 5 web documents of a question.
use class Summarizer's summarize method to obtain summary.
- input: article **(string)**
- output: [summary,article] **(list of strings)**

## 3. For Reranking of articles of a question
### 3.1. rerank.py
rerank.py is a python script that reranks articles of a question with additional information such as ...
use class Reranker's rerank method to obtain reranked articles.
- input: question **(string)**
- output: [article #1, article #2, ...] **(list of strings)**