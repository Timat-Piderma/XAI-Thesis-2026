# Code example taken from SHAP repository applied to my dataset
# https://github.com/shap/shap/blob/master/notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.ipynb

import pandas as pd
import matplotlib.pylab as pl
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

import shap

# Load Dataset
print("Loading dataset...")
df = pd.read_csv('Loan_default.csv')

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
xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
pl.savefig('xgboost.plot_importance.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'xgboost.plot_importance.png'.")
pl.show()

xgboost.plot_importance(model, importance_type="cover")
pl.title('xgboost.plot_importance(model, importance_type="cover")')
pl.savefig('xgboost.plot_importance_cover.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'xgboost.plot_importance_cover.png'.")
pl.show()

xgboost.plot_importance(model, importance_type="gain")
pl.title('xgboost.plot_importance(model, importance_type="gain")')
pl.savefig('xgboost.plot_importance_gain.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'xgboost.plot_importance_gain.png'.")
pl.show()

# Explain predictions
# Using Tree SHAP to explain the entire dataset
print("Generating Explainer (This may take a while)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize a single prediction
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=True)

# Visualize many (1000) predictions
shap.force_plot(explainer.expected_value, shap_values[:1000, :], X.iloc[:1000, :])

# Bar chart of mean importance
shap.summary_plot(shap_values, X, plot_type="bar")

# SHAP Summary Plot
shap.summary_plot(shap_values, X)