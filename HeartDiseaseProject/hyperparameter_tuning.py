# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load preprocessed dataset
df = pd.read_csv('DF1_preprocessed.csv')  # Make sure the file exists in the same directory

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grids
param_grids = {
    'lr': {
        'solver': ['lbfgs', 'liblinear'],
        'C': [0.01, 0.1, 1, 10]
    },
    'svc': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'dt': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gb': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'subsample': [0.8, 1.0]
    }
}

# Initialize models
models = {
    'lr': LogisticRegression(random_state=42, max_iter=1000),
    'svc': SVC(probability=True, random_state=42),
    'dt': DecisionTreeClassifier(random_state=42),
    'rf': RandomForestClassifier(random_state=42),
    'gb': GradientBoostingClassifier(random_state=42)
}

# Perform hyperparameter tuning
best_params = {}
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[model_name],
        n_iter=8,  # Number of random combinations to try
        scoring='accuracy',  # Metric to optimize
        cv=5,  # 5-fold cross-validation
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    search.fit(X_train, y_train)
    best_params[model_name] = search.best_params_
    print(f"Best parameters for {model_name}: {search.best_params_}\n")

# Print all best parameters
print("\nBest Parameters for Each Model:")
for model_name, params in best_params.items():
    print(f"{model_name}: {params}")