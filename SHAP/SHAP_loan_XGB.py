# Code example taken from SHAP repository applied to my dataset
# https://github.com/shap/shap/blob/master/notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.ipynb

import pandas as pd
from pathlib import Path
import matplotlib.pylab as pl
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

import shap

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

print("Applying One-Hot Encoding...")
# Convert categorical attributes with dummy numerical values
X = pd.get_dummies(df.drop(columns=COLS_TO_DROP), drop_first=True)
y = df[TARGET_COL]

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

# Training the model
params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
}

model = xgboost.train(
    params,
    d_train,
    5000,
    evals=[(d_test, "test")],
    verbose_eval=100,
    early_stopping_rounds=20,
)

# Classic feature attributions
output_path = script_folder / 'xgb_output/xgb_plot_importance.png'
xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

output_path = script_folder / 'xgb_output/xgb_plot_importance_cover.png'
xgboost.plot_importance(model, importance_type="cover")
pl.title('xgboost.plot_importance(model, importance_type="cover")')
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

output_path = script_folder / 'xgb_output/xgb_plot_importance_gain.png'
xgboost.plot_importance(model, importance_type="gain")
pl.title('xgboost.plot_importance(model, importance_type="gain")')
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# Explain predictions
# Using Tree SHAP to explain the entire dataset
print("Generating Explainer (This may take a while)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize a single prediction
output_path = script_folder / 'xgb_output/shap_xgb_force_plot_single.png'
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# Visualize many (1000) predictions
output_path = script_folder / 'xgb_output/shap_xgb_force_plot_interactive.html'
interactive_plot = shap.force_plot(explainer.expected_value, shap_values[:1000, :], X.iloc[:1000, :])
shap.save_html(str(output_path), interactive_plot)
print(f"Plot saved as '{output_path}'")

# Bar chart of mean importance
output_path = script_folder / 'xgb_output/shap_xgb_summary_bar.png'
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# SHAP Summary Plot
output_path = script_folder / 'xgb_output/shap_xgb_summary_dots.png'
shap.summary_plot(shap_values, X, show=False)
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")