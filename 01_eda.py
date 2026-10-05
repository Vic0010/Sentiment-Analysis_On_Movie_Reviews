import pandas as pd

# Load dataset
df = pd.read_csv("Dataset_IMDB/IMDB Dataset.csv")

print("Original dataset shape:", df.shape)

# Remove exact duplicate reviews
df = df.drop_duplicates(subset="review").reset_index(drop=True)

print("Dataset shape after removing duplicates:", df.shape)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Reviews After Cleaning:")
print(df["review"].duplicated().sum())
