import os
import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, PrecisionRecallDisplay
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
X_tv, X_test, y_tv, y_test = train_test_split(
    list(df.embedding.values), df.label, test_size=0.2, random_state=23
)
X_train, X_validate, y_train, y_validate = train_test_split(
        X_tv, y_tv, test_size=0.2, random_state=23
)
#classification
logreg = LogisticRegression(max_iter=1000, random_state=1919, solver='sag')
logreg.fit(X_train, y_train)
lr_predictions = logreg.predict(X_validate)
lr_vali_acc = logreg.score(X_validate, y_validate)
lr_precision = precision_score(y_validate, lr_predictions, average=None)
lr_recall = recall_score(y_validate, lr_predictions, average=None)
print("\tLogReg Vali-Acc: {:.3}".format(lr_vali_acc))
print(lr_precision)
print(lr_recall)

clf = RandomForestClassifier(n_estimators=100, criterion="gini")
clf.fit(X_train, y_train)
rf_predictions = clf.predict(X_validate)
rf_vali_acc = clf.score(X_validate, y_validate)
rf_precision = precision_score(y_validate, rf_predictions, average=None)
rf_recall = recall_score(y_validate, rf_predictions, average=None)
print("\tRandomForest Vali-Acc: {:.3}".format(rf_vali_acc))
print(rf_precision)
print(rf_recall)
'''
display = PrecisionRecallDisplay.from_estimator(
    clf, X_test, y_test, name="RandomForest"
)
display.show()
'''
#zero shot classification