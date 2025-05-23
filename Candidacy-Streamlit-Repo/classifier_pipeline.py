import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Step 1: Create a synthetic ordinal dataset
np.random.seed(42)

# Generating synthetic features
n_patients = 50
n_features = 10
X = np.random.rand(n_patients * 2, n_features)  # 2 entries per patient

# Generating ordinal scores with 20 possible values
y = np.random.randint(1, 21, size=n_patients * 2)  # "score" target with 20 ordinal classes

# Generating patient_ids (2 entries per patient)
patient_ids = np.repeat(np.arange(1, n_patients + 1), 2)

# Creating the dataframe
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
df['score'] = y
df['patient_id'] = patient_ids

# Step 2: Create the train-test split preserving the patient_id
X = df.drop(columns=['score', 'patient_id'])
y = df['score']
groups = df['patient_id']  # Grouping by patient_id

# Step 3: Train-test split using GroupShuffleSplit to preserve patient_id
train_size = 0.8  # 80% training data, 20% testing data
splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Step 4: Set up the pipeline with a RandomForestClassifier and StandardScaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 5: Set up the GridSearchCV with hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Step 6: Train the model using GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Step 7: Display the best parameters from the grid search
print(f"Best parameters: {grid_search.best_params_}")

# Step 8: Evaluate the model on the test data
y_pred = grid_search.predict(X_test)
print(f"Accuracy on test data: {accuracy_score(y_test, y_pred):.4f}")


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Create a small example dataset with 6 samples and a 'patient_id' grouping
data = {
    'feature_1': [1, 2, 3, 4, 5, 6],
    'feature_2': [7, 8, 9, 10, 11, 12],
    'score': [1, 2, 3, 4, 5, 6],
    'patient_id': [1, 1, 2, 2, 3, 3]  # Grouping by patient_id (2 samples per patient)
}

df = pd.DataFrame(data)

# Define the features and target
X = df[['feature_1', 'feature_2']]
y = df['score']
groups = df['patient_id']  # Group by patient_id

# Use GroupShuffleSplit to split the dataset while preserving the groups
splitter = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

# Split the data
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Show the train and test data
print("Training data:")
print(X_train)
print(y_train)

print("\nTest data:")
print(X_test)
print(y_test)
