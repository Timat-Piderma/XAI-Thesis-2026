from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm

np.random.seed(42)

# Loading Dataset
script_folder = Path(__file__).parent
dataset_path = script_folder.parent / 'data' / 'Loan_default.csv'
print("Loading dataset...")
df = pd.read_csv(dataset_path)

# Defining target column
TARGET_COL = 'Default'

# Pre-Processing
# Removing rows with missing values
df = df.dropna()

# Dropping 'LoanID' as it's an irrelevant identifier
df = df.drop(columns=['LoanID'])

# Splitting features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y       
)

print(f"Train Size: {X_train.shape}")
print(f"Test Size: {X_test.shape}")

categorical_columns = [
    'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
    'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

print("Applying Ordinal Encoding...")

encoder = sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])

categorical_features = [X_train.columns.get_loc(col) for col in categorical_columns]
categorical_names = {i: list(cats) for i, cats in enumerate(encoder.categories_)}

# Train Baseline
model = RandomForestClassifier(
    n_estimators=0,
    warm_start=True,
    random_state=42
)

step = 1
tot = 100

for i in tqdm(range(step, tot + 1, step), desc="Training Random Forest..."):
    model.n_estimators = i
    model.fit(X_train, y_train)

print(f"Model Accuracy: {sklearn.metrics.accuracy_score(y_test, model.predict(X_test))}")

# Saving Random Forest model and data
rf = {
    'model': model,
    'X': X,
    'y': y,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': X_test.columns,
    'categorical_names': categorical_names,
    'categorical_cols': X.select_dtypes(include=['object', 'category']).columns
}

joblib.dump(rf, script_folder / 'loan_rf.joblib')
print(f"Model and data saved as: {script_folder / 'loan_rf.joblib'}")