from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Step 2: Load the dataset
df = pd.read_csv('DF1.csv')  # Replace with your actual dataset file path

# Step 3: Define features and target
X = df.drop('target', axis=1)  # Features (drop target column)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Impute Missing Values
imputer = SimpleImputer(strategy='median')  # Use median imputation for numerical columns
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X.columns)


# Step 2: Apply Feature Transformation (e.g., log transformation)
X_train_imputed['age'] = np.log1p(X_train_imputed['age'])
X_test_imputed['age'] = np.log1p(X_test_imputed['age'])

# Step 3: Feature Selection using RFE
model = RandomForestClassifier(random_state=42)
selector = RFE(model, n_features_to_select=10)  # Choose top 10 features
X_train_selected = selector.fit_transform(X_train_imputed, y_train)
X_test_selected = selector.transform(X_test_imputed)

# Step 4: Train the Model using the selected features
model.fit(X_train_selected, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test_selected)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
