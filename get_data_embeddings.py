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

input_datapath = "../../nobu-data/s23-thesis-paper/labeled_body_shuffled_ds_0.csv" 
df = pd.read_csv(input_datapath, index_col=0)
print(df.head())
target_params = []

#cleaning to isolate the param entry as its own (should work)
with tqdm.tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        pbar.update(1)
        for i in range(10):
            current_param_entry = row["a" + str(i)]
            if current_param_entry != "ignore" and current_param_entry != "u":
                target_params.append(current_param_entry)
#this should now contain target_param as the parameter we're looking at
df['target_param'] = target_params
print(df.head())

embeddings_df = pd.DataFrame()
embeddings_df['param'] = df['target_param']

#using OpenAI tokeniser because this is how they'd like us to use it
#https://github.com/openai/tiktoken
'''
encoding = tiktoken.get_encoding(embedding_encoding)
# This may take a few minutes
embeddings_df["embedding"] = df['target_param'].apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("nobu-data/s23-thesis-paper/fine_food_reviews_with_embeddings_1k.csv")
'''