import pandas as pd
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import csv


TRAIN_OPTION = "3"
FILE_TRAIN1 = "Data/spam/spam_train1.csv"
FILE_TRAIN2 = "Data/spam/spam_train2.csv"
FILE_TEST = "Data/spam/spam_test.csv"
MODEL = "2"
TRAINING_DATA = True


def label_finder(path):
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        row = next(reader, None)

        for col, value in enumerate(row):
            if value.strip().lower() == "ham" or value.strip().lower() == "spam":
                return col


def load_data(path):
    label_col = label_finder(path)
    text_col = label_col + 1
    df = pd.read_csv(
        path, usecols=[label_col, text_col], names=["label", "text"], header=0
    )
    return df


if TRAIN_OPTION == "1":
    print("Using training dataset 1 only...")
    train = load_data(FILE_TRAIN1)

elif TRAIN_OPTION == "2":
    print("Using training dataset 2 only...")
    train = load_data(FILE_TRAIN2)

elif TRAIN_OPTION.lower() == "3":
    print("Using both training datasets...")
    train1 = load_data(FILE_TRAIN1)
    train2 = load_data(FILE_TRAIN2)
    train = pd.concat([train1, train2], axis=0).reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(
    train["text"],
    train["label"],
    test_size=0.2,
    shuffle=True,
)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"\b\w+\b",  # Only Letters and Numbers can exist
    min_df=2,  # Ensure there are at least 2 uses of the text
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
if MODEL == "1":
    clf = MultinomialNB()

elif MODEL == "2":
    clf = ComplementNB()


clf.fit(X_train_vec, y_train)

if TRAINING_DATA:
    text = X_test
    vec = X_test_vec
    labels = y_test
else:
    spam_test_df = pd.read_csv(FILE_TEST, header=0)
    text = spam_test_df.iloc[:, 0]
    vec = vectorizer.transform(text)
    labels = None

# Predict
pred = clf.predict(vec)

print(f"Accuracy:, {accuracy_score(labels, pred) *100:.2f}%")
print(classification_report(labels, pred ,zero_division=0))

# Export predictions
results = pd.DataFrame({"Prediction": pred})
results.replace({"spam": 1, "ham": 0}).to_csv("VuSpam.txt", index=False, header=False)
