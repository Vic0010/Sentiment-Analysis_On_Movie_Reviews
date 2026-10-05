import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("Dataset_IMDB/IMDB_cleaned.csv")

df = df.drop_duplicates(
    subset="clean_review"
).reset_index(drop=True)


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
# 4. TF-IDF with unigram + bigram
# --------------------------------------------------

print("Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Original number of features:",
      X_train_tfidf.shape[1])


# --------------------------------------------------
# 5. Feature selection experiments
# --------------------------------------------------

feature_counts = [5000, 10000, 20000]

results = []


for k in feature_counts:

    print("\n" + "=" * 60)
    print(f"TOP {k} FEATURES")
    print("=" * 60)

    # Select top k features
    selector = SelectKBest(
        score_func=chi2,
        k=k
    )

    X_train_selected = selector.fit_transform(
        X_train_tfidf,
        y_train
    )

    X_test_selected = selector.transform(
        X_test_tfidf
    )

    print("Selected features:",
          X_train_selected.shape[1])

    # Train SVM
    svm = LinearSVC(C=1.0)

    svm.fit(
        X_train_selected,
        y_train
    )

    # Predict
    y_pred = svm.predict(
        X_test_selected
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    results.append({
        "Features": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# --------------------------------------------------
# 6. Display final comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("FEATURE SELECTION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))
