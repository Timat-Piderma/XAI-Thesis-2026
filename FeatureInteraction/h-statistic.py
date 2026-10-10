from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import skexplain
import itertools
import numpy as np
import joblib

np.random.seed(42)

# Loading Model-Dataset Bundle
script_folder = Path(__file__).parent
bundle_path = script_folder.parent / 'models' / 'loan_rf.joblib'

print(f"Loading bundle from {bundle_path}...")
bundle = joblib.load(bundle_path)

rf = bundle['model']
X_test = bundle['X_test']
y_test = bundle['y_test']
feature_names = bundle['feature_names']
categorical_names = bundle['categorical_names']
categorical_cols = bundle['categorical_cols']

category_names = [None] * len(feature_names)

for i, col_name in enumerate(categorical_cols):
    c = X_test.columns.get_loc(col_name)
    category_names[c] = categorical_names[i]

explainer = skexplain.ExplainToolkit(("Random Forest", rf), X=X_test, y=y_test)

cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 
    'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 
    'LoanPurpose', 'HasCoSigner'
]

# Generate features combinations
features = list(itertools.combinations(cols, 2))
# Compute 1D ALE for all features and 2D ALE for the pairs
ale_1d_ds = explainer.ale(
    features='all', n_bootstrap=1, subsample=0.25, n_jobs=1, n_bins=20
)

ale_2d_ds = explainer.ale(
    features=features, n_bootstrap=1, subsample=0.25,
    n_jobs=-1, n_bins=20,
)

# Compute the H-statistic
hstat_results = explainer.friedman_h_stat(ale_1d_ds, ale_2d_ds, features=features)
     
fig = explainer.plot_importance(
    data=[hstat_results],
    panels=[('hstat', 'Random Forest')]
)

# Saving plot
output_path = script_folder / 'h-stat.png'
plt.savefig(output_path)
print(f"Plot saved as '{output_path}'")