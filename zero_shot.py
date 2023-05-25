import os
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity, get_embedding
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report

filepath = "../../nobu-data/s23-thesis-paper/test1k_0_lkf_param_embeddings.csv"

def label_score(review_embedding, label_embeddings):
            return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

def zero_shot_class(data_filepath):
    openai.api_key = os.getenv("S23PARAML")
    #load data
    df = pd.read_csv(data_filepath, index_col=0)
    print(df.head())

    #https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb
    label_embeddings = [get_embedding(label, engine="text-embedding-ada-002") for label in ["True", "False"]]

    #probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    probas = []
    for embedding in list(df.embedding.values):
            probas.append(label_score(embedding, label_embeddings))
    preds = probas.apply(lambda x: 'True' if x>0 else 'False')

    report = classification_report(df.label, preds)
    print(report)

zero_shot_class(filepath)