from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import xgboost
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
df = df.dropna()
df = df.drop(columns=['LoanID'])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y       
)

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

class TqdmCallback(xgboost.callback.TrainingCallback):
    def __init__(self, total_estimators):
        self.total_estimators = total_estimators
        self.pbar = None

    def before_training(self, model):                       
        self.pbar = tqdm(total=self.total_estimators, desc="Training XGBoost...")
        return model

    def after_iteration(self, model, epoch, evals_log):     
        self.pbar.update(1)
        return False

    def after_training(self, model):                        
        if self.pbar:
            self.pbar.close()
        return model

n_estimators=100

model = xgboost.XGBClassifier(
    n_estimators=n_estimators,
    max_depth=5,
    random_state=42,
    callbacks=[TqdmCallback(total_estimators=n_estimators)]
)

X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

model.fit(X_train_np, y_train_np)

print(f"Model Accuracy: {sklearn.metrics.accuracy_score(y_test_np, model.predict(X_test_np))}")
model.set_params(callbacks=None)

gbtree = {
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

joblib.dump(gbtree, script_folder / 'loan_xgb.joblib')
print(f"Model and data saved as: {script_folder / 'loan_xgb.joblib'}")