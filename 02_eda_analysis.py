import pandas as pd
import matplotlib.pyplot as plt
import re

# Load dataset
df = pd.read_csv("Dataset_IMDB/IMDB Dataset.csv")

# Remove duplicate reviews
df = df.drop_duplicates(subset="review").reset_index(drop=True)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 1. Review length analysis
# --------------------------------------------------

df["word_count"] = df["review"].apply(lambda x: len(x.split()))
df["character_count"] = df["review"].apply(len)

print("\nReview Length Statistics:")
print(df["word_count"].describe())

print("\nAverage words per review:",
      round(df["word_count"].mean(), 2))

print("Average characters per review:",
      round(df["character_count"].mean(), 2))

# --------------------------------------------------
# 2. Check HTML tags
# --------------------------------------------------

html_pattern = r"<[^>]+>"

html_reviews = df["review"].apply(
    lambda x: bool(re.search(html_pattern, x))
)

print("\nReviews containing HTML tags:", html_reviews.sum())
print("Percentage:",
      round(html_reviews.mean() * 100, 2), "%")

# --------------------------------------------------
# 3. Sentiment distribution
# --------------------------------------------------

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# --------------------------------------------------
# 4. Sample positive and negative reviews
# --------------------------------------------------

print("\nExample Positive Review:")
print(df[df["sentiment"] == "positive"]["review"].iloc[0])

print("\nExample Negative Review:")
print(df[df["sentiment"] == "negative"]["review"].iloc[0])

# --------------------------------------------------
# 5. Plot sentiment distribution
# --------------------------------------------------

plt.figure(figsize=(6, 4))

df["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("sentiment_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 6. Plot review length distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["word_count"], bins=50)

plt.title("Distribution of Review Length")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("review_length_distribution.png", dpi=300)
plt.show()
