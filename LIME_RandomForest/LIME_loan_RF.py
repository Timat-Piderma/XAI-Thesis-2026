from __future__ import print_function
from pathlib import Path
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
import numpy as np
import lime
import lime.lime_tabular
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

categorical_columns = [
    'Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 
    'HasMortgage', 'HasDependents', 'HasCoSigner'
]

print("Applying Label Encoding for LIME...")
categorical_names = {}
categorical_features = []

# Label Encoding for categorical features
for i, col in enumerate(X.columns):
    if col in categorical_columns or X[col].dtype == object or X[col].dtype == bool:
        categorical_features.append(i)
        le = sklearn.preprocessing.LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str)) 
        categorical_names[i] = le.classes_.tolist()

# Train Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y       
)

print(f"Train Size: {X_train.shape}")
print(f"Test Size: {X_test.shape}")

# Convert dataframes in NumPy array to avoid LIME warnings
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# Train Baseline
rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=0,
    warm_start=True,
    random_state=42
)

step = 1
tot = 100

for i in tqdm(range(step, tot + 1, step), desc="Training Random Forest..."):
    rf.n_estimators = i
    rf.fit(X_train_np, y_train_np)

print(f"Model Accuracy: {sklearn.metrics.accuracy_score(y_test_np, rf.predict(X_test_np))}")

# Create the Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_np, 
    feature_names=X_train.columns.tolist(), 
    class_names=['No Default', 'Default'], 
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    discretize_continuous=True, 
    mode='classification'
)

# Explaining an instance
i = np.random.randint(0, X_test_np.shape[0])

exp = explainer.explain_instance(
    X_test_np[i], 
    rf.predict_proba,  
    top_labels=1
)

# Save plot
output_path = script_folder / 'lime_rf_single_explanation.html'
exp.save_to_file(file_path=output_path, show_table=True, show_all=False)
print(f"Plot saved as '{output_path}'")