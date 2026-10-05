import pandas as pd
import re
import html


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("Dataset_IMDB/IMDB Dataset.csv")

print("Original dataset:", df.shape)


# --------------------------------------------------
# 2. Remove duplicate reviews
# --------------------------------------------------

df = df.drop_duplicates(subset="review").reset_index(drop=True)

print("After removing duplicates:", df.shape)


# --------------------------------------------------
# 3. Text preprocessing function
# --------------------------------------------------

def clean_text(text):

    # Convert to string
    text = str(text)

    # Decode HTML entities
    # Example: &amp; -> &
    text = html.unescape(text)

    # Remove HTML tags such as <br />
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Keep letters, numbers, whitespace and apostrophes
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 4. Apply preprocessing
# --------------------------------------------------

df["clean_review"] = df["review"].apply(clean_text)


# --------------------------------------------------
# 5. Display examples
# --------------------------------------------------

print("\nOriginal Review:")
print(df["review"].iloc[0])

print("\nCleaned Review:")
print(df["clean_review"].iloc[0])


# --------------------------------------------------
# 6. Check for empty reviews
# --------------------------------------------------

empty_reviews = (df["clean_review"].str.len() == 0).sum()

print("\nEmpty reviews after preprocessing:", empty_reviews)


# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

output_file = "Dataset_IMDB/IMDB_cleaned.csv"

df[["clean_review", "sentiment"]].to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_file)

print("\nFinal dataset shape:", df.shape)
