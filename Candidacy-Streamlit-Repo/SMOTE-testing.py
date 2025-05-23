import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE


X, y = make_classification(n_classes=5, class_sep=2, weights=[0.05, 0.15, 0.4, 0.3, 0.1],
                           n_informative=5, n_redundant=1, flip_y=0,
                           n_features=10, n_clusters_per_class=1, n_samples=2000, random_state=42)

# Check class distribution before SMOTE
print("Class distribution before SMOTE:", np.bincount(y))


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE on multi-class data
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:", np.bincount(y_train_resampled))

