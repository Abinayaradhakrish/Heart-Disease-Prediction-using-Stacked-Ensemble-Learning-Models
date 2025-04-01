import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
import joblib
import pickle
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Confusion Matrix




warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv('DF1.csv')  # Replace with your actual dataset path
X = data.drop('target', axis=1)  # Replace 'target' with the actual target column name
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best hyperparameters from your hyperparameter tuning
best_params = {
    'lr': {'solver': 'liblinear', 'C': 0.1, 'class_weight': 'balanced'},
    'svc': {'kernel': 'rbf', 'gamma': 'auto', 'C': 10, 'class_weight': 'balanced'},
    'dt': {'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None},
    'rf': {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None, 'class_weight': 'balanced'},
    'gb': {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.2}
}

# Stage 1 models (base learners) with best hyperparameters
base_models = [
    LogisticRegression(solver=best_params['lr']['solver'], C=best_params['lr']['C'], class_weight=best_params['lr']['class_weight']),
    SVC(probability=True, kernel=best_params['svc']['kernel'], gamma=best_params['svc']['gamma'], C=best_params['svc']['C'], class_weight=best_params['svc']['class_weight']),
    DecisionTreeClassifier(min_samples_split=best_params['dt']['min_samples_split'],
                           min_samples_leaf=best_params['dt']['min_samples_leaf'], max_depth=best_params['dt']['max_depth']),
    RandomForestClassifier(n_estimators=best_params['rf']['n_estimators'],
                           min_samples_split=best_params['rf']['min_samples_split'],
                           min_samples_leaf=best_params['rf']['min_samples_leaf'], max_depth=best_params['rf']['max_depth'], class_weight=best_params['rf']['class_weight']),
    XGBClassifier(subsample=best_params['gb']['subsample'], n_estimators=best_params['gb']['n_estimators'],
                  max_depth=best_params['gb']['max_depth'], learning_rate=best_params['gb']['learning_rate'], use_label_encoder=False, eval_metric='logloss')
]

# Meta-model
meta_model = LogisticRegression()

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stage 1: Generate predictions from base models
base_model_predictions = np.zeros((X_train.shape[0], len(base_models)))

# Train base models using cross-validation
fitted_base_models = []  # To store fitted models

for i, model in enumerate(base_models):
    fold_predictions = np.zeros(X_train.shape[0])
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_fold_train, y_fold_train)
        fold_predictions[val_idx] = model_clone.predict(X_fold_val)

    base_model_predictions[:, i] = fold_predictions

    # Fit the base model on the entire training data after cross-validation
    model.fit(X_train, y_train)
    fitted_base_models.append(model)  # Save the fully fitted model

# Stage 2: Train the meta-model (stacked model)
meta_model.fit(base_model_predictions, y_train)

# Save the meta-model and the fully fitted base models
stacked_model = {"meta_model": meta_model, "base_models": fitted_base_models}
with open("stacked_model.pkl", "wb") as f:
    pickle.dump(stacked_model, f)
print("Stacked model saved successfully!")

# Evaluate the stacked model
# Generate predictions for test data using base models
base_test_predictions = np.zeros((X_test.shape[0], len(fitted_base_models)))

for i, model in enumerate(fitted_base_models):
    base_test_predictions[:, i] = model.predict(X_test)

# Final predictions using the meta-model
y_pred = meta_model.predict(base_test_predictions)
y_proba = meta_model.predict_proba(base_test_predictions)[:, 1]

# Adjust threshold (e.g., 0.3 instead of 0.5) for better recall of minority class
new_threshold = 0.3
y_pred_new = (y_proba >= new_threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_new)
precision = precision_score(y_test, y_pred_new)
recall = recall_score(y_test, y_pred_new)
f1 = f1_score(y_test, y_pred_new)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred_new)

# Save evaluation results to a .pkl file
results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "confusion_matrix": cm
}
joblib.dump(results, "DF1_stacked_model_results.pkl")
print("Evaluation results saved successfully!")

# Print results
print("Stacked Model Evaluation with Threshold Adjustment:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Heart Disease"], yticklabels=["No Disease", "Heart Disease"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
print("Confusion Matrix:")
print(cm)
