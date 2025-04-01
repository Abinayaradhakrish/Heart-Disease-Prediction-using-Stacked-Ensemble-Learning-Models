# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
df = pd.read_csv('DF1.csv')

# Check unique values and counts in the target column
print("Unique values in the target column:", df['target'].unique())
print("\nValue counts for the target column:")
print(df['target'].value_counts())

# Step 1: Handle Missing Values
print("Before handling missing values:\n", df.isnull().sum())

# Replace missing values with 0 and then impute median values
df.fillna(0, inplace=True)
for column in df.columns:
    if df[column].dtype in ['float64', 'int64'] and column not in ['sex', 'target']:
        df[column] = df[column].replace(0, df[column].median())

print("\nAfter handling missing values:\n", df.isnull().sum())

# Step 2: Scale Features
print("\nScaling features...")
scaler = StandardScaler()
feature_columns = [col for col in df.columns if col not in ['target', 'sex']]
df[feature_columns] = scaler.fit_transform(df[feature_columns])
print("Feature scaling completed.\n")

# Save the scaler for use in Flask app
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'.")

# Step 3: Apply SMOTE to balance the dataset
print("\nApplying SMOTE to balance the dataset...")
X = df.drop('target', axis=1)
y = df['target']

# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new distribution after SMOTE
print("\nAfter SMOTE - Value counts for the target column:")
print(y_resampled.value_counts())

# Step 4: Check Correlation
print("\nAnalyzing correlations...")
correlation_matrix = pd.DataFrame(X_resampled).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 5: Save Preprocessed Data
preprocessed_df = pd.DataFrame(X_resampled, columns=X.columns)
preprocessed_df['target'] = y_resampled
preprocessed_df.to_csv('DF1_preprocessed_smote.csv', index=False)
print("Preprocessed data with SMOTE saved as 'DF1_preprocessed_smote.csv' successfully!")
