import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def IgnoreMissingData(X,y):
    # delete row with missing data
    X_train = X[~np.isnan(X).any(axis=1)]
    y_train = y[~np.isnan(X).any(axis=1)]
    return X_train,y_train
def ImputeMean(X,y):
    # Impute missing data with mean:
    iris_mean = np.nanmean(X, axis=0)
    X_train = np.where(np.isnan(X), iris_mean, X)
    Y_train=y
    return X_train,Y_train
def ImputeGlobalConstant(X,y):
    # Impute missing data with global constant 0:
    X_train = np.where(np.isnan(X), 0, X)
    Y_train = y
    return X_train,Y_train

def predict_label(X_train, y_train,X_test):
    # model
    gnb = GaussianNB().fit(X_train, y_train)
    # predict
    y_pred = gnb.predict(X_test)
    return y_pred
def main(prob):
    iris = pd.read_csv(f'IrisNan{prob}.csv',delimiter=',',header=0) # read csv file

    iris = np.array(iris) # convert to numpy

    # divide data into features and labels
    X = iris[:,0:4]
    y = iris[:,4]

    test_iris = pd.read_csv("Iris_test.csv",delimiter=',',header=0)
    test_iris = np.array(test_iris)
    X_test, y_test = test_iris[:,0:4], test_iris[:,4]

    X_train,y_train = IgnoreMissingData(X,y)
    y_pred = predict_label(X_train,y_train,X_test)
    print("Accuracy of Ignoring Missing Data: {}%".format(round(accuracy_score(y_pred,y_test)*100,2)))

    X_train,y_train = ImputeMean(X,y)
    y_pred = predict_label(X_train,y_train,X_test)
    print("Accuracy of Impute Mean Missing Data: {}%".format(round(accuracy_score(y_pred,y_test)*100,2)))

    X_train,y_train = ImputeGlobalConstant(X,y)
    y_pred = predict_label(X_train,y_train,X_test)
    print("Accuracy of Impute Global Constant Missing Data: {}%\n".format(round(accuracy_score(y_pred,y_test)*100,2)))

prob = 5
for i in range(4):
    print(f"Missing data {prob}%")
    main(prob)
    prob+=5
