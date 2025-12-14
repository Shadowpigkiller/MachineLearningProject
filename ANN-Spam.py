# ------------------
# 0 - Imports
# ------------------
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# ------------------
# 1 - Configuration
# ------------------

# set file paths & which to use for evaluation
TRAIN_OPTION = "both"   # "1", "2", or "both"
FILE_TRAIN1 = "spam_detection/spam_train1.csv"
FILE_TRAIN2 = "spam_detection/spam_train2.csv"
FILE_TEST = "spam_detection/spam_test.csv"
FILE_PREDICTIONS = "spam_detection/spam_test_predictions.csv"

# ANN parameters
LEARNING_RATE = 0.001
MAX_ITER = 50
HIDDEN_LAYERS = (150,)
ACTIVATION_FUNC = "relu"
SOLVER_ALGO = "adam"
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42

KFOLDS = 5

# ------------------
# 2 - Load the Data
# ------------------
def load_data(path):
    df = pd.read_csv(
        path,  # path to the file
        header=None,  # no column labels
        usecols=[0, 1],  # only first two columns contain necessary data
        names=["label", "text"],  # first column is data label, second column is email text
        encoding="latin-1",
        quotechar='"',  # prevents commas inside quotes from splitting data
        on_bad_lines="skip",
    )
    return df

def load_test_texts(path):
    """Load test texts robustly from 1- or 2-column CSVs.
    Why: test file schema may vary; this avoids hard failures."""
    df = pd.read_csv(
        path,
        header=None,
        encoding="latin-1",
        quotechar='"',
        on_bad_lines="skip",
    )
    if df.shape[1] >= 2:
        texts = df.iloc[:, 1]
    else:
        texts = df.iloc[:, 0]
    texts = texts.astype(str).str.strip()
    return texts

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
    train = pd.concat([train1, train2], axis=0).reset_index(drop=True)  # combine vertically and reindex values from -

# ----------------------
# 3 - Clean the Data
# ----------------------

# Prepare for vectorization
train = train.dropna(subset=["label", "text"])  # drop rows with missing labels or text
train["label"] = train["label"].astype(str).str.strip().str.lower()  # normalize labels
train["text"] = train["text"].astype(str).str.strip()  # normalize email text
train = train[train["label"].isin(["spam", "ham"])].copy()  # ensure labels are either 'spam' or 'ham'

# Encode labels numerically (force strict int64 and validate)
label_map = {"ham": 0, "spam": 1}
train.loc[:, "label"] = train["label"].map(label_map)

if train["label"].isna().any():
    raise ValueError("Found unknown labels outside {'ham','spam'} after mapping.")

train.loc[:, "label"] = train["label"].astype(np.int64)
y_array = train["label"].to_numpy(dtype=np.int64)

unique = np.unique(y_array)
if unique.size < 2 or not set(unique).issubset({0, 1}):
    raise ValueError(f"Labels must be binary {{0,1}} with both classes present. Got: {unique}")

print(f"Loaded {len(train)} samples: {int((y_array==0).sum())} ham, {int((y_array==1).sum())} spam")

# ----------------------
# 4 - K-Fold Cross-Validation
# ----------------------
print(f"\nRunning {KFOLDS}-Fold Stratified Cross-Validation...")
cv_pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=HIDDEN_LAYERS,
                activation=ACTIVATION_FUNC,
                solver=SOLVER_ALGO,
                learning_rate_init=LEARNING_RATE,
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

cv = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    cv_pipeline,
    train["text"],   # raw text (vectorized inside the pipeline)
    y_array,         # strict NumPy int64 array
    cv=cv,
    scoring="accuracy",
)

for i, s in enumerate(cv_scores, start=1):
    print(f" Fold {i} Accuracy: {round(s * 100, 5):.5f}%")
print(f" Mean CV Accuracy: {round(cv_scores.mean() * 100, 5):.5f}%")
print(f" Std CV Accuracy: {round(cv_scores.std(ddof=1) * 100, 5):.5f}%\n")

# ----------------------
# 5 - Build the Model
# ----------------------

X_train, X_test, y_train, y_test = train_test_split(
    train["text"], y_array, test_size=0.2, random_state=42, stratify=y_array
)

# Tfidf vectorization
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# build model
model = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation=ACTIVATION_FUNC,
    solver=SOLVER_ALGO,
    learning_rate_init=LEARNING_RATE,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
)

# ----------------------
# 6 - Train the Model
# ----------------------

print("Training model...")

y_train = np.asarray(y_train, dtype=np.int64)  # ensure integer dtype
y_test = np.asarray(y_test, dtype=np.int64)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {round(accuracy * 100, 5):.5f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ----------------------
# 7 - Export Predictions
# ----------------------

print("\nPreparing for final predictions...")

# Refit vectorizer on all training texts
final_vectorizer = TfidfVectorizer(
    stop_words="english", max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE
)
X_all_tfidf = final_vectorizer.fit_transform(train["text"])

# Refit the same model spec on all training data
final_model = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation=ACTIVATION_FUNC,
    solver=SOLVER_ALGO,
    learning_rate_init=LEARNING_RATE,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
)
final_model.fit(X_all_tfidf, y_array)

# Load test file
print(f"Generating predictions for external test file: {FILE_TEST}")
test_texts = load_test_texts(FILE_TEST)
X_test_external = final_vectorizer.transform(test_texts)
test_preds = final_model.predict(X_test_external).astype(np.int64)

# Save predictions:
pd.Series(test_preds, dtype=np.int64).to_csv(FILE_PREDICTIONS, index=False, header=False)
print(f"Saved predictions to: {FILE_PREDICTIONS}")
