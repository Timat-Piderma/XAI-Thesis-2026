from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
from sklearn.metrics import accuracy_score
from anchor import utils
from anchor import anchor_tabular
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

encoder = sklearn.preprocessing.OrdinalEncoder()

categorical_indices = [X.columns.get_loc(col) for col in categorical_columns]
categorical_names_dict = {}

for col in categorical_columns:
    idx = X.columns.get_loc(col)
    # Get the unique values of the column
    categorical_names_dict[idx] = X[col].unique().astype(str).tolist()

# Apply Encoding
X_encoded = X.copy()
X_encoded[categorical_columns] = encoder.fit_transform(X[categorical_columns])

# Convert data into numpy array
X_array = X_encoded.values
feature_names = X.columns.tolist()

# Train Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_array, y, 
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

print(f"Model Accuracy: {accuracy_score(y_test, rf.predict(X_test))}")

predict_fn = lambda x: rf.predict(x)

# Create Explainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names= ['No Default', 'Default'],
    feature_names= feature_names,
    train_data= X_train,
    categorical_names= categorical_names_dict
)

idx = np.random.randint(0, X_test.shape[0])

# Generate Explanation
print('Prediction: ', explainer.class_names[predict_fn(X_test[idx].reshape(1, -1))[0]])
exp = explainer.explain_instance(X_test[idx], predict_fn, threshold=0.95)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())