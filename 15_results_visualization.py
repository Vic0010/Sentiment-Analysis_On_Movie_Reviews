import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. MODEL ACCURACY COMPARISON
# ============================================================

models = [
    "Naive Bayes\nUnigram",
    "Naive Bayes\nUni + Bigram",
    "Logistic Regression\nUnigram",
    "Logistic Regression\nUni + Bigram",
    "SVM\nUnigram",
    "SVM\nUni + Bigram"
]

accuracy = [
    86.64,
    89.59,
    89.92,
    90.55,
    89.95,
    91.86
]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, accuracy)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Accuracy Comparison")
plt.ylim(80, 95)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("01_model_accuracy.png", dpi=300)
plt.show()


# ============================================================
# 2. F1-SCORE COMPARISON
# ============================================================

f1_scores = [
    86.51,
    89.57,
    90.10,
    90.77,
    90.08,
    91.99
]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, f1_scores)

plt.ylabel("F1-score (%)")
plt.xlabel("Model")
plt.title("F1-score Comparison")
plt.ylim(80, 95)

for bar, value in zip(bars, f1_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("02_f1_score_comparison.png", dpi=300)
plt.show()


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

feature_counts = [
    5000,
    10000,
    20000
]

feature_accuracy = [
    90.5910,
    91.0448,
    91.4482
]

plt.figure(figsize=(9, 6))

plt.plot(
    feature_counts,
    feature_accuracy,
    marker="o"
)

plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy (%)")
plt.title("Effect of Feature Selection on SVM Accuracy")
plt.ylim(89, 93)

for x, y in zip(feature_counts, feature_accuracy):
    plt.annotate(
        f"{y:.2f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )

plt.tight_layout()
plt.savefig("03_feature_selection.png", dpi=300)
plt.show()


# ============================================================
# 4. SVM HYPERPARAMETER TUNING
# ============================================================

C_values = [
    0.1,
    0.5,
    1,
    2,
    5
]

C_accuracy = [
    90.6817,
    91.8616,
    91.8616,
    91.8616,
    91.7910
]

plt.figure(figsize=(9, 6))

plt.plot(
    C_values,
    C_accuracy,
    marker="o"
)

plt.xlabel("SVM C Value")
plt.ylabel("Accuracy (%)")
plt.title("SVM Hyperparameter Tuning")
plt.ylim(89, 93)

for x, y in zip(C_values, C_accuracy):
    plt.annotate(
        f"{y:.2f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )

plt.tight_layout()
plt.savefig("04_svm_tuning.png", dpi=300)
plt.show()


# ============================================================
# 5. FINAL MODEL CONFUSION MATRIX
# ============================================================

confusion_matrix = np.array([
    [4478, 461],
    [346, 4631]
])

plt.figure(figsize=(7, 6))

plt.imshow(confusion_matrix)

plt.title("Confusion Matrix — Final SVM Model")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar(label="Number of Reviews")

plt.tight_layout()
plt.savefig("05_confusion_matrix.png", dpi=300)
plt.show()


# ============================================================
# 6. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print("Model       : Linear SVM")
print("Features    : TF-IDF Unigram + Bigram")
print("C           : 0.5")
print("Accuracy    : 91.86%")
print("Precision   : 90.95%")
print("Recall      : 93.05%")
print("F1-score    : 91.99%")
print("Features    : 613,727")

print("\nGraphs saved:")
print("01_model_accuracy.png")
print("02_f1_score_comparison.png")
print("03_feature_selection.png")
print("04_svm_tuning.png")
print("05_confusion_matrix.png")
