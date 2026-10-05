import pandas as pd
import numpy as np
import re
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = Path("Dataset_IMDB")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)


# =========================================================
# FIND DATASET
# =========================================================

csv_files = list(DATASET_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found inside Dataset_IMDB.")

print("Dataset found:", csv_files[0])

df = pd.read_csv(csv_files[0])

print("Original shape:", df.shape)


# =========================================================
# CLEAN DATA
# =========================================================

df = df.drop_duplicates(subset=["review"]).copy()

df = df.dropna(subset=["review", "sentiment"])

def clean_text(text):
    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["review"] = df["review"].apply(clean_text)

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

df = df.dropna(subset=["sentiment"])

print("After cleaning:", df.shape)


# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================

X = df["review"]
y = df["sentiment"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# =========================================================
# TF-IDF
# =========================================================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("Number of features:", X_train_tfidf.shape[1])


# =========================================================
# TRAIN SVM
# =========================================================

print("\nTraining SVM with C = 0.5...")

model = LinearSVC(C=0.5)

model.fit(X_train_tfidf, y_train)

print("Training complete.")


# =========================================================
# FIND BEST THRESHOLD
# =========================================================

print("\nFinding best decision threshold...")

val_scores = model.decision_function(X_val_tfidf)

thresholds = np.arange(-0.50, 0.51, 0.01)

best_threshold = 0
best_f1 = 0

results = []

for threshold in thresholds:

    val_predictions = (val_scores >= threshold).astype(int)

    accuracy = accuracy_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions)

    results.append((threshold, accuracy, f1))

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n===================================================")
print("THRESHOLD TUNING RESULTS")
print("===================================================")

print("Best threshold :", round(best_threshold, 4))
print("Validation F1  :", round(best_f1, 4))


# =========================================================
# TEST USING BEST THRESHOLD
# =========================================================

test_scores = model.decision_function(X_test_tfidf)

test_predictions = (
    test_scores >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions
)

print("\n===================================================")
print("TEST RESULTS WITH OPTIMIZED THRESHOLD")
print("===================================================")

print("Accuracy :", round(test_accuracy, 4))
print("F1-score :", round(test_f1, 4))


# =========================================================
# SAVE MODEL + VECTORIZER + THRESHOLD
# =========================================================

joblib.dump(
    model,
    MODEL_DIR / "threshold_svm.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "threshold_tfidf.pkl"
)

joblib.dump(
    best_threshold,
    MODEL_DIR / "sentiment_threshold.pkl"
)

print("\nSaved:")
print("models/threshold_svm.pkl")
print("models/threshold_tfidf.pkl")
print("models/sentiment_threshold.pkl")
