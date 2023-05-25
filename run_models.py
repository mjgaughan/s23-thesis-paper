import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

filepath = "../../nobu-data/s23-thesis-paper/test1k_0_lkf_param_embeddings.csv"

def linear_ots_class(embeddings_filepath):
    np.random.seed(100)
    #import data
    df = pd.read_csv(embeddings_filepath, index_col=0)
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
    #logreg
    logreg = LogisticRegression(max_iter=1000, random_state=1919, solver='sag')
    logreg.fit(X_train, y_train)
    lr_predictions = logreg.predict(X_validate)
    lr_vali_acc = logreg.score(X_validate, y_validate)
    print("\tLogReg Vali-Acc: {:.3}".format(lr_vali_acc))
    report = classification_report(y_validate, lr_predictions)
    print(report)
    #random forest 
    clf = RandomForestClassifier(n_estimators=100, criterion="gini")
    clf.fit(X_train, y_train)
    rf_predictions = clf.predict(X_validate)
    rf_vali_acc = clf.score(X_validate, y_validate)
    print("\tRandomForest Vali-Acc: {:.3}".format(rf_vali_acc))
    report = classification_report(y_validate, rf_predictions)
    print(report)

linear_ots_class(filepath)
