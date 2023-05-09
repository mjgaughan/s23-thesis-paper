import os
import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

openai.api_key = os.getenv("S23PARAML")
np.random.seed(100)
#import data
input_datapath = "../../nobu-data/s23-thesis-paper/test1k_0_lkf_param_embeddings.csv"
df = pd.read_csv(input_datapath, index_col=0)
print(df.head())
#turn string into array 
df["embedding"] = df.embedding.apply(eval).apply(np.array)  # convert string to array
#train, validate, test split 
train0, train1, validate, test = np.split(df, 4)
train = pd.concat([train0, train1])
print(len(train.index))
print(len(validate.index))
print(len(test.index))
#classification
'''
logreg = LogisticRegression(max_iter=1000, random_state=1841, solver='sag')
logreg.fit(train["embedding"], train["label"])
vali_acc = logreg.score(validate["embedding"], validate["label"])
print("\tVali-Acc: {:.3}".format(vali_acc))
'''
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train["embedding"], train["label"])
vali_acc = clf.score(validate["embedding"], validate["label"])
print("\tVali-Acc: {:.3}".format(vali_acc))
#TypeError: only size-1 arrays can be converted to Python scalars

#zero shot classification