import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# 1. Load dataset
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
# 4. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Number of features:", X_train_tfidf.shape[1])


# --------------------------------------------------
# 5. Create SVM
# --------------------------------------------------

svm = LinearSVC(C=0.5)


# --------------------------------------------------
# 6. Calibrate SVM
# --------------------------------------------------

print("\nCalibrating SVM...")
print("This may take some time.")

calibrated_model = CalibratedClassifierCV(
    estimator=svm,
    method="sigmoid",
    cv=3
)

calibrated_model.fit(
    X_train_tfidf,
    y_train
)

print("Calibration complete.")


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

y_pred = calibrated_model.predict(X_test_tfidf)

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

print("\n" + "=" * 60)
print("CALIBRATED SVM RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 8. Test probability predictions
# --------------------------------------------------

test_reviews = [
    "The movie was absolutely fantastic!",
    "The movie was boring and terrible.",
    "The movie was decent and the acting was good.",
    "The movie started well but became boring and predictable towards the end."
]

print("\n" + "=" * 60)
print("PROBABILITY TESTS")
print("=" * 60)

for review in test_reviews:

    # Clean using the same preprocessing approach
    import re
    import html

    cleaned = html.unescape(review)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-zA-Z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    vector = tfidf.transform([cleaned])

    prediction = calibrated_model.predict(vector)[0]

    probabilities = calibrated_model.predict_proba(vector)[0]

    classes = calibrated_model.classes_

    print("\nReview:")
    print(review)

    print("Prediction:", prediction.upper())

    for class_name, probability in zip(classes, probabilities):
        print(
            f"{class_name.capitalize()}: "
            f"{probability * 100:.2f}%"
        )


# --------------------------------------------------
# 9. Save calibrated model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    calibrated_model,
    "models/calibrated_sentiment_svm.pkl"
)

joblib.dump(
    tfidf,
    "models/calibrated_tfidf_vectorizer.pkl"
)

print("\nSaved:")
print("models/calibrated_sentiment_svm.pkl")
print("models/calibrated_tfidf_vectorizer.pkl")
