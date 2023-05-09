import os
import openai
import pandas as pd
import tiktoken
import tqdm

openai.api_key = os.getenv("S23PARAML")
#https://platform.openai.com/docs/guides/embeddings/use-cases

# embedding model parameters (https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb)
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

input_datapath = "../../nobu-data/s23-thesis-paper/0_lkernel_params_mutable.csv"
df = pd.read_csv(input_datapath, index_col=0)
print(df.head())
#using OpenAI tokeniser because this is how they'd like us to use it
#https://github.com/openai/tiktoken

embeddings_df = pd.DataFrame()
embeddings_df['param'] = df['target_param']
embeddings_df['label'] = df["not_written_to"]
#for testing_only grabbing top 1k
embeddings_df = embeddings_df.head(1000)
#tokenizing
encoding = tiktoken.get_encoding(embedding_encoding)
print(len(embeddings_df.index))
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# This may take a few minutes
embeddings_df["embedding"] = embeddings_df['param'].apply(lambda x: get_embedding(x, model=embedding_model))
embeddings_df.to_csv("../../nobu-data/s23-thesis-paper/test1k_0_lkf_param_embeddings.csv")
