import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
#read data
#Set 1
train_data1 = pd.read_csv("Data/classification/TrainData1.txt", sep='\t', header=None)
train_labels1 = pd.read_csv("Data/classification/TrainLabel1.txt", header=None).values.ravel()
test_data1 = pd.read_csv("Data/classification/TestData1.txt", sep='\t', header=None)

#Set 2
train_data2 = pd.read_csv("Data/classification/TrainData2.txt", sep='\t', header=None)
train_labels2 = pd.read_csv("Data/classification/TrainLabel2.txt", header=None).values.ravel()
test_data2 = pd.read_csv("Data/classification/TestData2.txt", sep='\t', header=None)
#train_data2 = train_data2.apply(pd.to_numeric, errors='coerce')
print(train_data2.dtypes)
#Set 3
train_data3 = pd.read_csv("Data/classification/TrainData3.txt", sep='\t', header=None)
train_labels3 = pd.read_csv("Data/classification/TrainLabel3.txt", header=None).values.ravel()
test_data3 = pd.read_csv("Data/classification/TestData3.txt", sep='\t', header=None)

#Set 4
train_data4 = pd.read_csv("Data/classification/TrainData4.txt", sep='\t', header=None)
train_labels4 = pd.read_csv("Data/classification/TrainLabel4.txt", header=None).values.ravel()
test_data4 = pd.read_csv("Data/classification/TestData4.txt", sep='\t', header=None)

#Set 1
#train_data1.replace(1.00000000000000e+99, np.nan, inplace=True)
#test_data1.replace(1.00000000000000e+99, np.nan, inplace=True)
#imputer = SimpleImputer(strategy='mean')
#train_data1 = pd.DataFrame(imputer.fit_transform(train_data1))
#test_data1 = pd.DataFrame(imputer.transform(test_data1))
#knn_no_scale = KNeighborsClassifier(n_neighbors=5)
#scores_no_scale = cross_val_score(knn_no_scale, train_data1, train_labels1, cv=5)
# Fit on all training data and predict
#knn_no_scale.fit(train_data1, train_labels1)
#pred_no_scale = knn_no_scale.predict(test_data1)
#pipe_scaled = Pipeline([
#    ('scaler', StandardScaler()),
#    ('knn', KNeighborsClassifier(n_neighbors=5))
#])
#scores_scaled = cross_val_score(pipe_scaled, train_data1, train_labels1, cv=5)
#pipe_scaled.fit(train_data1, train_labels1)
#pred_scaled = pipe_scaled.predict(test_data1)
#print(f"\nSet 1: KNN without scaling - Mean CV Accuracy: {scores_no_scale.mean():.4f}")
#print(f"Set 1: KNN with scaling - Mean CV Accuracy: {scores_scaled.mean():.4f}")

#Set 2
'''
train_data2.replace(1.00000000000000e+99, np.nan, inplace=True)
test_data2.replace(1.00000000000000e+99, np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
train_data2 = pd.DataFrame(imputer.fit_transform(train_data2))
test_data2 = pd.DataFrame(imputer.transform(test_data2))
knn_no_scale2 = KNeighborsClassifier(n_neighbors=5)
scores_no_scale2 = cross_val_score(knn_no_scale2, train_data2, train_labels2, cv=5)
# Fit on all training data and predict
knn_no_scale2.fit(train_data2, train_labels2)
pred_no_scale2 = knn_no_scale2.predict(test_data2)
pipe_scaled2 = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
scores_scaled2 = cross_val_score(pipe_scaled2, train_data2, train_labels2, cv=5)
pipe_scaled2.fit(train_data2, train_labels2)
pred_scaled2 = pipe_scaled2.predict(test_data2)
print(f"\nSet 2: KNN without scaling - Mean CV Accuracy: {scores_no_scale2.mean():.4f}")
print(f"Set 2: KNN with scaling - Mean CV Accuracy: {scores_scaled2.mean():.4f}")
'''