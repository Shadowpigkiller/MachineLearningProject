import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

train_data = pd.read_csv("Data/classification/TrainData1.txt", sep='\t', header=None)
train_labels = pd.read_csv("Data/classification/TrainLabel1.txt", header=None).values.ravel()
test_data = pd.read_csv("Data/classification/TestData1.txt", sep='\t', header=None)

print(f"Train shape: {train_data.shape}")
print(f"Test shape:  {test_data.shape}")
print(f"Unique classes: {np.unique(train_labels)}")


train_data.replace(1.00000000000000e+99, np.nan, inplace=True)
test_data.replace(1.00000000000000e+99, np.nan, inplace=True)

imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data))
test_data = pd.DataFrame(imputer.transform(test_data))


knn_no_scale = KNeighborsClassifier(n_neighbors=5)
scores_no_scale = cross_val_score(knn_no_scale, train_data, train_labels, cv=5)

# Fit on all training data and predict
knn_no_scale.fit(train_data, train_labels)
pred_no_scale = knn_no_scale.predict(test_data)


pipe_scaled = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

scores_scaled = cross_val_score(pipe_scaled, train_data, train_labels, cv=5)

pipe_scaled.fit(train_data, train_labels)
pred_scaled = pipe_scaled.predict(test_data)

print(f"\nKNN without scaling - Mean CV Accuracy: {scores_no_scale.mean():.4f}")
print(f"KNN with scaling - Mean CV Accuracy: {scores_scaled.mean():.4f}")