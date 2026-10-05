import effector
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

categorical_cols = X.select_dtypes(include=['object', 'category']).columns

encoder = sklearn.preprocessing.OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Train Test split (Ora X contiene solo numeri)
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

# Convert dataframes in NumPy array to avoid LIME warnings
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

for i in tqdm(range(step, tot + 1, step), desc="Training Random Forest..."):
    rf.n_estimators = i
    rf.fit(X_train_np, y_train_np)

print(f"Model Accuracy: {sklearn.metrics.accuracy_score(y_test_np, rf.predict(X_test_np))}")

predict = effector.adapters.classifier_proba(rf)  # a plain numpy -> numpy callable

category_names = [None] * len(X_train.columns)

for i, col_name in enumerate(categorical_cols):
    c = X_train.columns.get_loc(col_name)
    # Convertiamo i numpy array in liste standard per Effector
    category_names[c] = list(encoder.categories_[i])

schema = effector.Schema(
    feature_names=X_train.columns,
    feature_types=[
        "ordinal", "ordinal", "ordinal", "ordinal", "ordinal", "ordinal",
        "ordinal", "ordinal", "ordinal", "nominal", "nominal", "nominal",
        "nominal", "nominal", "nominal", "nominal"
    ],
    category_names=category_names,
    target_name="Default",
)

report = effector.explain(
    X_train_np, 
    predict, 
    y_train_np, 
    schema=schema, 
    nof_instances=100, 
    random_state=42
)

report.to_html(script_folder / "report.html")

ale = effector.ALE(
    X_train_np, 
    predict, 
    schema=schema, 
    nof_instances=100, 
    random_state=42
)

for i, col in enumerate(X_train.columns):
    ale.plot(i)
    plt.savefig(script_folder / f"ale_plot_{col}.png", bbox_inches='tight', dpi=300)
    plt.close()