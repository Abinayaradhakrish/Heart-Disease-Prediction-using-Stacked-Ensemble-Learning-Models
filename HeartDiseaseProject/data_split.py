# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the preprocessed dataset
df = pd.read_csv('DF1_preprocessed.csv')  # Replace with the correct file path if necessary
print("Loaded preprocessed dataset:\n")
print(df.head())  # Display the first few rows to confirm the data is loaded correctly

# Step 2: Separate features (X) and target (y)
X = df.drop(columns=['target'])  # Drop the target column to create feature matrix
y = df['target']  # Extract the target column

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Use stratify to maintain class distribution
)

# Display the sizes of the training and testing sets
print(f"\nDataset split completed:")
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# Verify the class distribution in the training and testing sets
print("\nClass distribution in the training set:")
print(y_train.value_counts())
print("\nClass distribution in the testing set:")
print(y_test.value_counts())
