# === ELITE COLAB TRAINING SCRIPT ===
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# 1. Load Data
# (Make sure you dragged 'Telco-Customer-Churn.csv' into the Colab file sidebar first!)
df = pd.read_csv("Telco-Customer-Churn.csv")

# 2. robust Data Engineering
# Fix TotalCharges (coerce errors to NaN, then fill with 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Define X (Features) and y (Target)
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify Column Types automatically
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Constructing the ML Pipeline
# This bundles preprocessing with the model so your App never crashes on "Male/Female" strings.

# Preprocessing for Numerical Data (Scale it)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for Categorical Data (Encode it)
# handle_unknown='use_encoded_value' makes it crash-proof against new categories
categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 4. Grid Search Optimization (The "Top 1%" Part)
# We define a "Base Pipeline"
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# We define the hyperparameter grid to search
param_grid = {
    'classifier__n_estimators': [100, 200, 300],  # Try different forest sizes
    'classifier__max_depth': [10, 20, None],      # Try different tree depths
    'classifier__min_samples_split': [2, 5, 10],  # Control overfitting
    'classifier__min_samples_leaf': [1, 2, 4]     # Smooth the decision boundaries
}

print(" Starting Hyperparameter Tuning on Colab Cloud CPUs...")
# n_jobs=-1 uses ALL Cores (Safe in Colab, fast!)
grid_search = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=5,                 # 5-Fold Cross Validation (very robust)
    scoring='f1',         # Optimize for F1 Score (Balance of Precision/Recall)
    n_jobs=-1,
    verbose=1
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit the Grid Search
grid_search.fit(X_train, y_train)

# 5. Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n===  Best Model Performance ===")
print(f"Best Params: {grid_search.best_params_}")
print(classification_report(y_test, y_pred))

# 6. Save & Download
joblib.dump(best_model, "model_v1.pkl")
print(" Model saved as 'model_v1.pkl'. Please download it from the files sidebar!")