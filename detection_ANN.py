## ------------------
## 0 - Imports
## ------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

## ------------------
## 1 - Configuration
## ------------------

# set file paths & which to use for evaluation
TRAIN_OPTION = "1"   # "1", "2", or "both"
FILE_TRAIN1 = "spam_detection/spam_train1.csv"
FILE_TRAIN2 = "spam_detection/spam_train2.csv"

# ANN parameters
LEARNING_RATE = 0.001
MAX_ITER = 40
HIDDEN_LAYERS = (150,)
ACTIVATION_FUNC = "relu"
SOLVER_ALGO = "adam"
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42


## ------------------
## 2 - Load the Data
## ------------------
def load_data(path):
    df = pd.read_csv(
        path, # path to the file
        header=None, # no column labels
        usecols=[0, 1], # only first two columns contain necessary data
        names=["label", "text"], # first column is data label, second column is email text
        encoding="latin-1", #
        quotechar='"', # prevents commas inside quotes from splitting data
        on_bad_lines="skip"
    )
    return df

if TRAIN_OPTION == "1":
    print("Using training dataset 1 only...")
    train = load_data(FILE_TRAIN1)

elif TRAIN_OPTION == "2":
    print("Using training dataset 2 only...")
    train = load_data(FILE_TRAIN2)

elif TRAIN_OPTION.lower() == "both":
    print("Using both training datasets...")
    train1 = load_data(FILE_TRAIN1)
    train2 = load_data(FILE_TRAIN2)
    train = pd.concat([train1, train2], axis=0).reset_index(drop=True) # combine vertically and reindex values from -

## ----------------------
## 3 - Clean the Data
## ----------------------

# Prepare for vectorization
train = train.dropna(subset=["label", "text"]) # drop rows with missing labels or text
train["label"] = train["label"].astype(str).str.strip().str.lower() # make all labels lowercase strings with no spaces
train["text"] = train["text"].astype(str).str.strip() # make all email text strings with no spaces
train = train[train["label"].isin(["spam", "ham"])].copy() # make fresh copy, ensuring laebls are either 'spam' or 'ham'

# Encode labels numerically (force integer type)
train.loc[:, "label"] = train["label"].map({"ham": 0, "spam": 1}).astype(int) # convert laebls to 0 or 1 (1 = true -> spam is detected)

print(f"Loaded {len(train)} samples: {sum(train['label']==0)} ham, {sum(train['label']==1)} spam") # print number of samples loaded and their true labels

## ----------------------
## 4 - Build the Model
## ----------------------

X_train, X_test, y_train, y_test = train_test_split(
    train["text"], train["label"], test_size=0.2, random_state=42, stratify=train["label"]
) # split data into a training and testing data set (80% for training, 20% for testing)

# build Tfidf vectorization -> "Term Frequency Inverse Document Frequency"
vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE) # ignores unhelpful english words, only tracks most up to top 5,000 most informative words, looks at 1 and 2 word phrases
X_train_tfidf = vectorizer.fit_transform(X_train) # analyze whole training vocab and convert individual test emails into vectors
X_test_tfidf = vectorizer.transform(X_test) # analyze whole testing vocab and convert individual train emails into vectors

# build model 
model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, activation=ACTIVATION_FUNC, solver=SOLVER_ALGO,
                      learning_rate_init=LEARNING_RATE, max_iter=MAX_ITER, random_state=RANDOM_STATE)

# contains 1 hidden layer with 150 neurons (can add more with tuple)
# uses ReLU activation function and adam optimizer (like gradient descent)
# learning rate of 0.001, maximum interactions of 40 (even if doesn't fully converge)

## ----------------------
## 5 - Train the Model
## ----------------------

print("Training model...")

y_train = np.array(y_train, dtype=int) # convert training data to numpy array
y_test = np.array(y_test, dtype=int) # convert testing data to numpy array

model.fit(X_train_tfidf, y_train) # fit model (vectorized email input, training label output)

y_pred = model.predict(X_test_tfidf) # produce array of guesses for each training sample

accuracy = accuracy_score(y_test, y_pred) # compare predicted to observed labels in training data
print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n") # print resulting accuracy rate

print("Classification Report:") 
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"])) # print report for model results

# NOTES:
# model has high test accuracy, so i'm unsure what improvement steps to make
# i was considering bootstrapping (basically making multiple networks based off different test data)
# bootstrapping is usually used for simpler models like trees; might be expensive for ANNs
# looking into novel ways to do emsembling / resampling without being crazy long to compute