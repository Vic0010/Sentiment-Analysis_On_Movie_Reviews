import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("Dataset_IMDB/IMDB_cleaned.csv")

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
# 4. Display sizes
# --------------------------------------------------

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Check class distribution
# --------------------------------------------------

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())
