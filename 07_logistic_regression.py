import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("Dataset_IMDB/IMDB_cleaned.csv")

df = df.drop_duplicates(
    subset="clean_review"
).reset_index(drop=True)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and labels
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
# Evaluation function
# --------------------------------------------------

def evaluate_model(model, X_train_features, X_test_features, name):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train_features, y_train)

    # Predict
    y_pred = model.predict(X_test_features)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="positive"
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


# ==================================================
# EXPERIMENT 1 — UNIGRAM
# ==================================================

print("\nCreating unigram TF-IDF...")

tfidf_uni = TfidfVectorizer(
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_uni = tfidf_uni.fit_transform(X_train)
X_test_uni = tfidf_uni.transform(X_test)

print("Unigram features:", X_train_uni.shape[1])


lr_uni = LogisticRegression(
    max_iter=1000,
    C=1.0
)

lr_uni = evaluate_model(
    lr_uni,
    X_train_uni,
    X_test_uni,
    "LOGISTIC REGRESSION — UNIGRAM"
)


# ==================================================
# EXPERIMENT 2 — UNIGRAM + BIGRAM
# ==================================================

print("\nCreating unigram + bigram TF-IDF...")

tfidf_bi = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_bi = tfidf_bi.fit_transform(X_train)
X_test_bi = tfidf_bi.transform(X_test)

print("Unigram + bigram features:", X_train_bi.shape[1])


lr_bi = LogisticRegression(
    max_iter=1000,
    C=1.0
)

lr_bi = evaluate_model(
    lr_bi,
    X_train_bi,
    X_test_bi,
    "LOGISTIC REGRESSION — UNIGRAM + BIGRAM"
)
