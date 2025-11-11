import pandas as pd
import re
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer


TRAIN_OPTION = "1" # 2 Has issues right now due to Column issues
FILE_TRAIN1 = "Data/spam/spam_train1.csv"
FILE_TRAIN2 = "Data/spam/spam_train2.csv"



def load_data(path):
    # column =[]
    # if TRAIN_OPTION == 1:
    #     column = [0,2]
    # else
    
    df = pd.read_csv(
        path,
        usecols=[0, 1],
        names=["label", "text"],
        header=None
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
    train = pd.concat([train1, train2], axis=0).reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split( 
    train["text"],
    train["label"],
    test_size=0.2,
    random_state=42,
    shuffle=True,
)


# Cleans the Data
# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
#     text = re.sub(r"\S+@\S+", " ", text)
#     text = re.sub(r"[^a-z\s]", " ", text)
#     tokens = text.split()
#     return [t for t in tokens if len(t) > 2]


# vectorizer = CountVectorizer(
#     analyzer=clean,
#     token_pattern=None,
#     min_df=2,
# )

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"\b\w+\b",   # Only Letters and Numbers can exist
    min_df=2,                   # Ensure there are at least 2 uses of the text
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Model
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Evaluate
pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
