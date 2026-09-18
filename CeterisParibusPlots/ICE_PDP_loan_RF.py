from __future__ import print_function
from pathlib import Path
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
import numpy as np
import dalex as dx
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

# We must remove 'LoanID' (or any unique identifier) to avoid MemoryError.
COLS_TO_DROP = [TARGET_COL, 'LoanID']

print("Applying Label Encoding...")
X = df.drop(columns=COLS_TO_DROP)
y = df[TARGET_COL]

categorical_columns = [
    'Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 
    'HasMortgage', 'HasDependents', 'HasCoSigner'
]

for col in X.columns:
    if col in categorical_columns or X[col].dtype == object or X[col].dtype == bool:
        le = sklearn.preprocessing.LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))


# Train Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y       
)

print(f"Train Size: {X_train.shape}")
print(f"Test Size: {X_test.shape}")

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
    rf.fit(X_train, y_train)

def predict_function(model, data):
    return rf.predict_proba(data)[:, 1]

print(f"Model Accuracy: {sklearn.metrics.accuracy_score(y_test, rf.predict(X_test))}")

# Create the Explainer
exp = dx.Explainer(rf, data=X, y=y,  predict_function=predict_function, label='Random Forest')

pdp = exp.model_profile(
    variables=['Age','Income','LoanAmount','CreditScore'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_1.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['MonthsEmployed','NumCreditLines','InterestRate','LoanTerm'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_2.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['DTIRatio','Education','EmploymentType','MaritalStatus'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_3.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['HasMortgage','HasDependents','LoanPurpose','HasCoSigner'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_4.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")