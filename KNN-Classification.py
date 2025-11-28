import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def runModel(train_data, test_data, train_labels):
    knn_no_scale = KNeighborsClassifier(n_neighbors=5)
    scores_no_scale = cross_val_score(knn_no_scale, train_data, train_labels, cv=3)
    # Fit on all training data and predict
    knn_no_scale.fit(train_data, train_labels)
    pred_no_scale = knn_no_scale.predict(test_data)
    pipe_scaled = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    scores_scaled = cross_val_score(pipe_scaled, train_data, train_labels, cv=3)
    pipe_scaled.fit(train_data, train_labels)
    pred_scaled = pipe_scaled.predict(test_data)
    print(f"KNN without scaling - Mean CV Accuracy: {scores_no_scale.mean():.4f}")
    print(f"KNN with scaling - Mean CV Accuracy: {scores_scaled.mean():.4f}")

#Read data
#Set 1
train_data1 = np.loadtxt("Data/classification/TrainData1.txt", dtype=float)
test_data1 = np.loadtxt("Data/classification/TestData1.txt", dtype=float)
train_labels1 = np.loadtxt("Data/classification/TrainLabel1.txt", dtype=int)

#Set 2
train_data2 = np.loadtxt("Data/classification/TrainData2.txt", dtype=float)
test_data2 = np.loadtxt("Data/classification/TestData2.txt", dtype=float)
train_labels2 = np.loadtxt("Data/classification/TrainLabel2.txt", dtype=int)

#Set 3
train_data3 = np.loadtxt("Data/classification/TrainData3.txt", dtype=float)
test_data3 = np.loadtxt("Data/classification/TestData3.txt", dtype=float)
train_labels3 = np.loadtxt("Data/classification/TrainLabel3.txt", dtype=int)

#Set 4
train_data4 = np.loadtxt("Data/classification/TrainData4.txt", dtype=float)
test_data4 = np.loadtxt("Data/classification/TestData4.txt", dtype=float)
train_labels4 = np.loadtxt("Data/classification/TrainLabel4.txt", dtype=int)

#Set 1
print("Training and Testing Set 1")
train_data1[train_data1 == 1e99] = np.nan
col_means = np.nanmean(train_data1, axis=0)
inds = np.where(np.isnan(train_data1))
train_data1[inds] = np.take(col_means, inds[1])

test_data1[test_data1 == 1e99] = np.nan
col_means = np.nanmean(test_data1, axis=0)
inds = np.where(np.isnan(test_data1))
test_data1[inds] = np.take(col_means, inds[1])
runModel(train_data=train_data1, test_data=test_data1, train_labels=train_labels1)

#Set 2
print("\nTraining and Testing Set 2")
train_data2[train_data2 == 1e99] = np.nan
col_means = np.nanmean(train_data2, axis=0)
inds = np.where(np.isnan(train_data2))
train_data2[inds] = np.take(col_means, inds[1])

test_data2[test_data2 == 1e99] = np.nan
col_means = np.nanmean(test_data2, axis=0)
inds = np.where(np.isnan(test_data2))
test_data2[inds] = np.take(col_means, inds[1])
runModel(train_data=train_data2, test_data=test_data2, train_labels=train_labels2)

#Set 3
print("\nTraining and Testing Set 3")
train_data3[train_data3 == 1e99] = np.nan
col_means = np.nanmean(train_data3, axis=0)
inds = np.where(np.isnan(train_data3))
train_data3[inds] = np.take(col_means, inds[1])

test_data3[test_data3 == 1e99] = np.nan
col_means = np.nanmean(test_data3, axis=0)
inds = np.where(np.isnan(test_data3))
test_data3[inds] = np.take(col_means, inds[1])
runModel(train_data=train_data3, test_data=test_data3, train_labels=train_labels3)

#Set 4
print("\nTraining and Testing Set 4")
train_data4[train_data4 == 1e99] = np.nan
col_means = np.nanmean(train_data4, axis=0)
inds = np.where(np.isnan(train_data4))
train_data4[inds] = np.take(col_means, inds[1])

test_data4[test_data4 == 1e99] = np.nan
col_means = np.nanmean(test_data4, axis=0)
inds = np.where(np.isnan(test_data4))
test_data4[inds] = np.take(col_means, inds[1])
runModel(train_data=train_data4, test_data=test_data4, train_labels=train_labels4)