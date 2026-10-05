import re
import html
import joblib


# --------------------------------------------------
# 1. Load saved model and TF-IDF vectorizer
# --------------------------------------------------

model = joblib.load("models/sentiment_svm.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# --------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------

def clean_text(text):

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Keep letters, numbers, spaces and apostrophes
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 3. Prediction function
# --------------------------------------------------

def predict_sentiment(review):

    # Clean review
    cleaned_review = clean_text(review)

    # Convert to TF-IDF
    review_vector = tfidf.transform([cleaned_review])

    # Predict
    prediction = model.predict(review_vector)[0]

    # Get decision score
    score = model.decision_function(review_vector)[0]

    return prediction, score


# --------------------------------------------------
# 4. Interactive prediction
# --------------------------------------------------

print("=" * 60)
print("       MOVIE REVIEW SENTIMENT ANALYZER")
print("=" * 60)

print("\nType a movie review.")
print("Type 'exit' to stop.\n")


while True:

    review = input("Enter review: ")

    if review.lower() == "exit":
        print("\nExiting...")
        break

    prediction, score = predict_sentiment(review)

    print("\nPrediction:", prediction.upper())
    print("Decision score:", round(score, 4))

    print("-" * 60)
