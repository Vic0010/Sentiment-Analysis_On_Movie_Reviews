import re

def normalize_text(text):
    """
    Normalize text consistently during training and prediction.
    In particular, treat '&' as the word 'and'.
    """
    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize ampersand to the natural-language equivalent
    text = re.sub(r"\s*&\s*", " and ", text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
