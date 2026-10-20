import pymupdf
import re
import os
from sentence_transformers import SentenceTransformer

def extract_text(file_path):
    text=''
    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            text+=page.get_text()+'\n'
    return text.strip()

import joblib
model=joblib.load('resume_screening_model.pkl')

import os
folder_path=r"C:\Users\hites\Downloads\archive (18)\data\data\ACCOUNTANT"
resume_text=[]
resume_name=[]

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.pdf'):
        file_path=os.path.join(folder_path,filename)

        try:
            text=extract_text(file_path)

            if text:
                resume_text.append(text)
                resume_name.append(filename)
        except Exception as e:
            print('Error reading {filename} : {e}')


predictions = model.predict(resume_text)

for name,categories in zip(resume_name,predictions):
    print('Resume:',name)
    print('Predicted Categories:',categories)
    print('-'*40)


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


job_description = """
Accountant with experience in accounting, finance, bookkeeping, and financial
management. Responsible for maintaining financial records, preparing financial
statements and reports, managing general ledger accounts, recording journal
entries, reconciling bank and account balances, processing accounts payable
and accounts receivable, handling invoices and payments, and supporting
month-end and year-end close.

Experienced in budgeting, forecasting, auditing, taxation, payroll, compliance,
and financial analysis. Skilled in Microsoft Excel and accounting software
such as QuickBooks, SAP, Oracle, Tally, or ERP systems. Knowledge of GAAP,
accounting principles, balance sheets, income statements, and cash flow reports.
"""
docoments=[job_description]+resume_text


from sentence_transformers import SentenceTransformer 
model=SentenceTransformer("all-MiniLM-L6-v2")
embedding=model.encode(docoments,batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True)
job_description_embedding=embedding[0:1]
resume_text_embedding=embedding[1:]

similarity_score=cosine_similarity(resume_text_embedding,job_description_embedding).flatten()

result=pd.DataFrame({'Resume':resume_name,'predicted_category':predictions,'scores':similarity_score*100})
result=result.sort_values(by='scores',ascending=False).reset_index(drop=True)
print(result.head())
























































































































# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     ngram_range=(1, 2),
#     max_features=20000,
#     sublinear_tf=True
# )
# tfidf_metrics=vectorizer.fit_transform(docoments)

# job_vectors=tfidf_metrics[0:1]
# resume_vectors=tfidf_metrics[1:]

# similarity_score=cosine_similarity(job_vectors,resume_vectors).flatten()


# result=pd.DataFrame({'Resume':resume_name,'Similarity_score':similarity_score*100})

# result=result.sort_values(by='Similarity_score',ascending=False).reset_index(drop=True)

# print(result.head())


# print(result.head(10))
# print(result[["Resume", "Similarity_score", "Category"]].head(10))
# print(result["Similarity_score"].describe())  