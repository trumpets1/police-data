import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


def cnb_process(df, test_size=0.33):
    encoder = OrdinalEncoder()
    X = df.iloc[:, [1, 2, 3, 4]]
    y = df.iloc[:, -1]
    X['CALLTYPE_CODE'] = encoder.fit_transform(X[['CALLTYPE_CODE']])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=142
    )
    return (X_train, X_test, y_train, y_test)


def cnb_predictor(X_train, y_train):
    cnb = CategoricalNB()
    cnb.fit(X_train, y_train)
    return cnb


def oc_svm_predictor(X_train, anomaly_fraction):
    svm = SGDOneClassSVM(
        nu=anomaly_fraction,
        shuffle=True,
        fit_intercept=True,
        random_state=42,
        tol=1e-6,)
    svm.fit(X_train)
    return svm
