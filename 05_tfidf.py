import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("Dataset_IMDB/IMDB_cleaned.csv")

# Remove duplicates
df = df.drop_duplicates(subset="clean_review").reset_index(drop=True)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate reviews and labels
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


# ==================================================
# EXPERIMENT 1: UNIGRAMS
# ==================================================

print("\n========== UNIGRAM TF-IDF ==========")

tfidf_unigram = TfidfVectorizer(
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_uni = tfidf_unigram.fit_transform(X_train)
X_test_uni = tfidf_unigram.transform(X_test)

print("Training matrix shape:", X_train_uni.shape)
print("Testing matrix shape:", X_test_uni.shape)

print("Number of unigram features:",
      len(tfidf_unigram.get_feature_names_out()))


# ==================================================
# EXPERIMENT 2: UNIGRAMS + BIGRAMS
# ==================================================

print("\n========== UNIGRAM + BIGRAM TF-IDF ==========")

tfidf_bigram = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_bi = tfidf_bigram.fit_transform(X_train)
X_test_bi = tfidf_bigram.transform(X_test)

print("Training matrix shape:", X_train_bi.shape)
print("Testing matrix shape:", X_test_bi.shape)

print("Number of unigram + bigram features:",
      len(tfidf_bigram.get_feature_names_out()))


# --------------------------------------------------
# 4. Display some vocabulary
# --------------------------------------------------

print("\nSample unigram features:")
print(tfidf_unigram.get_feature_names_out()[:20])

print("\nSample unigram + bigram features:")
print(tfidf_bigram.get_feature_names_out()[:20])
