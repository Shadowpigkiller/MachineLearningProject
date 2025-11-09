## ------------------
## 0 - Imports
## ------------------
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


## ------------------
## 1 - Configuration
## ------------------

# Set file paths
TRAIN_DATA_PATH = "classification/TrainData1.txt"
TRAIN_LABEL_PATH = "classification/TrainLabel1.txt"

# Decision Tree parameters
CRITERION = "gini"
MAX_DEPTH = None
RANDOM_STATE = 42

# Test size ratio
TEST_SIZE = 0.2


## ------------------
## 2 - Load the Data
## ------------------

print("Loading training data...")

X = np.loadtxt(TRAIN_DATA_PATH)
y = np.loadtxt(TRAIN_LABEL_PATH, dtype=int)

print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")


## ----------------------
## 3 - Clean the Data
## ----------------------

# Replace missing values (1e99) with column means
print("Handling missing values...")

X[X == 1e99] = np.nan
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_means, inds[1])

print("Missing values replaced with column means.")


## ----------------------
## 4 - Split Train/Test
## ----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


## ----------------------
## 5 - Train the Model
## ----------------------

print("Training Decision Tree model...")

clf = DecisionTreeClassifier(
    criterion=CRITERION,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE
)
clf.fit(X_train, y_train)

print("Model training complete.")


## ----------------------
## 6 - Evaluate the Model
## ----------------------

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Results")
print("----------------------")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))


## ----------------------
## 7 - Interactive Check Mode
## ----------------------

while True:
    cmd = input('\nType "check" to inspect an observation, or "exit" to quit: ').strip().lower()

    if cmd == "exit":
        print("Exiting program.")
        break

    elif cmd == "check":
        try:
            index = int(input(f"Enter observation index (0–{len(X_test) - 1}): "))
            if index < 0 or index >= len(X_test):
                print("Index out of range.")
                continue

            features = X_test[index]
            true_label = y_test[index]
            predicted = clf.predict([features])[0]

            print(f"\nObservation #{index}")
            print("-------------------------")
            print("Features:", np.round(features, 3))
            print(f"Predicted label: {predicted}")
            print(f"True label:      {true_label}")

        except ValueError:
            print("Please enter a valid integer index.")

    else:
        print('Unknown command. Type "check" or "exit".')
