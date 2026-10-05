import streamlit as st
import joblib
import re
from pathlib import Path

from text_preprocessing import normalize_text


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# =========================================================
# MODEL FILES
# =========================================================

MODEL_PATH = Path("models/normalized_svm.pkl")
VECTORIZER_PATH = Path("models/normalized_tfidf.pkl")
THRESHOLD_PATH = Path("models/normalized_threshold.pkl")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, vectorizer, threshold


try:

    model, vectorizer, threshold = load_model()

except Exception as e:

    st.error("Could not load the trained model files.")
    st.code(str(e))
    st.stop()


# =========================================================
# VALIDATE USER INPUT
# =========================================================

def validate_review(review):

    text = review.strip()

    # Empty input
    if not text:
        return False, "Please enter a movie review."

    # -----------------------------------------------------
    # IMPORTANT:
    # Validate ORIGINAL text BEFORE normalization.
    # This prevents inputs such as $$$&&&& from becoming
    # valid after '&' is converted to 'and'.
    # -----------------------------------------------------

    original_words = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?",
        text
    )

    original_letters = re.findall(
        r"[A-Za-z]",
        text
    )

    # At least 3 actual words
    if len(original_words) < 3:

        return False, (
            "Please enter a meaningful movie review "
            "with at least 3 words."
        )

    # -----------------------------------------------------
    # Check percentage of alphabetic characters
    # -----------------------------------------------------

    total_chars = len(
        re.sub(r"\s+", "", text)
    )

    if total_chars == 0:

        return False, "Please enter a meaningful movie review."

    alphabet_ratio = (
        len(original_letters) / total_chars
    )

    # Reject symbol/number-heavy input
    if alphabet_ratio < 0.50:

        return False, (
            "This does not look like a valid movie review. "
            "Please enter meaningful text."
        )

    # -----------------------------------------------------
    # ONLY NOW normalize the text
    # -----------------------------------------------------

    clean = normalize_text(text)

    return True, clean


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #a0a7b5;
        margin-bottom: 30px;
    }

    .positive-box {

        background-color: #123d2b;
        color: #35e58b;

        padding: 22px;

        border-radius: 10px;

        text-align: center;

        font-size: 27px;

        font-weight: bold;
    }

    .negative-box {

        background-color: #48272b;
        color: #ff7373;

        padding: 22px;

        border-radius: 10px;

        text-align: center;

        font-size: 27px;

        font-weight: bold;
    }

    .invalid-box {

        background-color: #432326;
        color: #ff8b8b;

        padding: 22px;

        border-radius: 10px;

        text-align: center;

        font-size: 23px;

        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">'
    'MOVIE REVIEW SENTIMENT ANALYZER'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter a movie review and the machine learning model '
    'will predict whether the sentiment is positive or negative.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# USER INPUT
# =========================================================

review = st.text_area(
    "Enter your movie review:",
    placeholder=(
        "Example: The movie was absolutely fantastic. "
        "The acting was brilliant..."
    ),
    height=180
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("Analyze Sentiment"):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    valid, processed_review = validate_review(review)

    if not valid:

        st.markdown(
            '<div class="invalid-box">'
            'INVALID REVIEW'
            '</div>',
            unsafe_allow_html=True
        )

        st.warning(processed_review)

    else:

        # -------------------------------------------------
        # TF-IDF TRANSFORMATION
        # -------------------------------------------------

        X = vectorizer.transform(
            [processed_review]
        )

        # -------------------------------------------------
        # SVM DECISION SCORE
        # -------------------------------------------------

        score = float(
            model.decision_function(X)[0]
        )

        # -------------------------------------------------
        # OPTIMIZED THRESHOLD
        # -------------------------------------------------

        if score >= threshold:

            sentiment = "POSITIVE"

        else:

            sentiment = "NEGATIVE"


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.markdown("## Prediction")


        if sentiment == "POSITIVE":

            st.markdown(
                '<div class="positive-box">'
                'POSITIVE'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="negative-box">'
                'NEGATIVE'
                '</div>',
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        st.write(
            f"**Decision score:** `{score:.4f}`"
        )

        st.write(
            f"**Optimized threshold:** `{threshold:.2f}`"
        )


        # -------------------------------------------------
        # UNCERTAINTY WARNING
        # -------------------------------------------------

        if abs(score - threshold) < 0.10:

            st.info(
                "This review is close to the model's "
                "decision boundary, so the prediction "
                "is relatively uncertain."
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("---")

st.markdown("## Model Information")


col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Algorithm:** Linear Support Vector Machine"
    )

    st.write(
        "**Features:** TF-IDF Unigrams + Bigrams"
    )

    st.write(
        "**SVM C:** 0.5"
    )


with col2:

    st.write(
        "**Threshold:** Validation-optimized"
    )

    st.write(
        "**Text normalization:** HTML + symbol normalization"
    )

    st.write(
        "**Dataset:** IMDB Movie Reviews"
    )
