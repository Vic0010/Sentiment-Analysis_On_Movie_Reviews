# Sentiment-Analysis_On_Movie_Reviews
Machine learning-based sentiment analysis of IMDB movie reviews using TF-IDF, Naive Bayes, Logistic Regression, and Linear SVM, with an interactive Streamlit web application.

## Overview

This project implements a machine learning-based sentiment analysis system for movie reviews. The system predicts whether a given movie review expresses a **positive** or **negative** sentiment.

Multiple machine learning algorithms and text feature representations were evaluated, and the best-performing model was integrated into an interactive Streamlit web application.

## Dataset

The project uses the **IMDB Movie Reviews Dataset**, containing 50,000 movie reviews divided into two sentiment classes:

- Positive
- Negative

The dataset was explored and cleaned before model training. Duplicate reviews were removed and text preprocessing was performed to prepare the reviews for machine learning.

## Project Workflow

The project follows these steps:

1. Exploratory Data Analysis
2. Data Cleaning
3. Train-Test Split
4. Text Preprocessing
5. TF-IDF Feature Extraction
6. Unigram and Bigram Feature Representation
7. Model Training
8. Feature Selection
9. SVM Hyperparameter Tuning
10. Negation Handling Experiment
11. Threshold Optimization
12. Final Model Training
13. Sentiment Prediction
14. Streamlit Web Application

## Machine Learning Models

The following classification algorithms were evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

Both **TF-IDF Unigrams** and **TF-IDF Unigrams + Bigrams** were tested.

## Model Comparison

| Model | Features | Accuracy |
|---|---|---:|
| Naive Bayes | Unigram | 86.64% |
| Naive Bayes | Unigram + Bigram | 89.59% |
| Logistic Regression | Unigram | 89.92% |
| Logistic Regression | Unigram + Bigram | 90.55% |
| Linear SVM | Unigram | 89.95% |
| Linear SVM | Unigram + Bigram | **91.86%** |

The **Linear SVM with TF-IDF Unigrams + Bigrams** achieved the highest accuracy.

## Final Model

The final model uses:

- **Algorithm:** Linear Support Vector Machine
- **Features:** TF-IDF Unigrams + Bigrams
- **SVM C:** 0.5
- **Test Accuracy:** 91.86%
- **F1-score:** 91.99%

The trained model and TF-IDF vectorizer are stored in the `models/` directory.

## Feature Selection

Feature selection was evaluated using different numbers of selected TF-IDF features.

The tested feature sizes were:

- 5,000
- 10,000
- 20,000

The results showed that increasing the number of selected features improved performance, with 20,000 features achieving an accuracy of approximately 91.45%.

## SVM Hyperparameter Tuning

Different values of the SVM regularization parameter `C` were tested:

- 0.1
- 0.5
- 1
- 2
- 5

The best configuration was:

```text
C = 0.5
Accuracy = 91.86%
F1-score = 91.99%
