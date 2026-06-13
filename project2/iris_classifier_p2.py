import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for overview
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")
print(f"\nFeature statistics:\n{df.describe()}")

# 2. Split First, Then Scale (Fixes Pitfalls 1 & 3)
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True,   # Pitfall 3 fixed
    stratify=y      # Pitfall 3 fixed
)

print(f"\n=== SPLIT RESULT ===")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# Pitfall 1 & 2 fixed: fit_transform on train, ONLY transform on test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nPre-scaling  mean of feature 0: {X_train[:, 0].mean():.4f}")
print(f"Post-scaling mean of feature 0: {X_train_scaled[:, 0].mean():.4f}")

# 3. Initial Model Evaluation
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Pitfall 2 fixed: predicting on scaled test data
y_predictions = knn_model.predict(X_test_scaled)

print(f"\n=== K=5 INITIAL RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_predictions) * 100:.2f}%")

# 4. The Elbow Curve (Fixes Pitfall 4)
error_rates = []
k_range = range(1, 31)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, y_pred_temp))

plt.figure(figsize=(12, 5))
plt.plot(
    k_range,
    error_rates,
    color='blue',
    linestyle='--',
    marker='o',
    markerfacecolor='red',
    markersize=8
)

plt.title('KNN Elbow Curve: Error Rate vs. K Value', fontsize=14)
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150)
plt.show()

print("\n→ Elbow curve saved as 'elbow_curve.png'")

# 5. Final Model Validation (Fixes Pitfall 5)
OPTIMAL_K = 7

knn_final = KNeighborsClassifier(n_neighbors=OPTIMAL_K)
knn_final.fit(X_train_scaled, y_train)

y_final_predictions = knn_final.predict(X_test_scaled)

print("\n" + "=" * 55)
print(f"  FINAL VALIDATION REPORT (K={OPTIMAL_K})")
print("=" * 55)

final_accuracy = accuracy_score(y_test, y_final_predictions)
print(f"\nOverall Accuracy: {final_accuracy * 100:.2f}%")

# Pitfall 5 fixed: reporting per-class metrics
print(f"\n--- Per-Class Report ---")
print(
    classification_report(
        y_test,
        y_final_predictions,
        target_names=iris.target_names
    )
)

cm = confusion_matrix(y_test, y_final_predictions)

print(f"\n--- Confusion Matrix (Raw) ---")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title(f'Confusion Matrix — KNN (K={OPTIMAL_K})', fontsize=13)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\n→ Confusion matrix saved as 'confusion_matrix.png'")