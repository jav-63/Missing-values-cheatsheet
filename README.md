import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Create a dummy production-style dataset
data = {
    'age': [22, np.nan, 35, 45, np.nan, 28, 54, 40],
    'income': [50000, 80000, np.nan, 120000, 60000, np.nan, 140000, 90000],
    'city': ['New York', 'Paris', 'New York', np.nan, 'London', 'Paris', 'London', np.nan],
    'tier': ['Premium', 'Free', np.nan, 'Premium', 'Free', 'Free', 'Premium', 'Free'],
    'churn': [0, 1, 0, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['churn'])
y = df['churn']

# Split data FIRST to ensure zero data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define feature groups by data type and missingness pattern
num_skewed_cols = ['income']          # Needs median imputation due to outliers
num_normal_cols = ['age']             # Needs KNN imputation for complex patterns
cat_cols = ['city', 'tier']           # Categorical columns with missing text values

# 3. Build sub-pipelines for each feature type
skewed_transformer = Pipeline(steps=[
    # add_indicator=True tracks the missingness pattern automatically
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler())
])

normal_transformer = Pipeline(steps=[
    # Advanced multivariate neighbor imputation
    ('imputer', KNNImputer(n_neighbors=3, add_indicator=True)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # fill_value='Missing' creates an explicit category for NaNs
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 4. Combine transformers into a single global ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_skewed', skewed_transformer, num_skewed_cols),
        ('num_normal', normal_transformer, num_normal_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop' # Explicitly drops any unlisted columns for security
)

# 5. Create the master end-to-end production model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Fit the pipeline (Learns imputation values and scales from Training Set ONLY)
pipeline.fit(X_train, y_train)

# 7. Predict safely (Applies learned training rules to Test Set without leakage)
predictions = pipeline.predict(X_test)
print("Pipeline executed successfully. Test predictions:", predictions)
