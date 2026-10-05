import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("Dataset_IMDB/IMDB_cleaned.csv")

df = df.drop_duplicates(
    subset="clean_review"
).reset_index(drop=True)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate text and labels
# --------------------------------------------------

X = df["clean_review"]
y = df["sentiment"]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Create TF-IDF vectorizer
# --------------------------------------------------

print("\nCreating TF-IDF vectorizer...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# --------------------------------------------------
# 5. Fit TF-IDF on training data
# --------------------------------------------------

X_train_tfidf = tfidf.fit_transform(X_train)

print("Number of features:", X_train_tfidf.shape[1])


# --------------------------------------------------
# 6. Train final SVM
# --------------------------------------------------

print("\nTraining final SVM...")

model = LinearSVC(C=0.5)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training complete.")


# --------------------------------------------------
# 7. Create models directory
# --------------------------------------------------

import os

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# 8. Save model and vectorizer
# --------------------------------------------------

joblib.dump(
    model,
    "models/sentiment_svm.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


print("\nSaved:")
print("models/sentiment_svm.pkl")
print("models/tfidf_vectorizer.pkl")
