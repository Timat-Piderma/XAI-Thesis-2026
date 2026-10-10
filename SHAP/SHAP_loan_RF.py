import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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

# Convert dataframes in NumPy array to avoid compatability issues
X_test_np = X_test.values
y_test_np = y_test.values

# Shap Global Explanation
print("\nInitializing SHAP TreeExplainer...")

explainer = shap.TreeExplainer(rf)

# For performance reasons during development, we calculate SHAP values on a sample of the test set.
X_test_sample = X_test.sample(n=500, random_state=42)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test_sample)

# We select index 1, which represents the probability of the 'Default' class.
shap_values_for_default = shap_values[:, :, 1]

# Generate the Summary Plot
print("Generating SHAP Summary Plot...")
# The summary plot combines feature importance with feature effects.
# Each dot is a single prediction from the test sample.
shap.summary_plot(
    shap_values_for_default, 
    X_test_sample, 
    show=False # Set to False so we can customize or save the plot before showing it
)

# Save and display
output_path = script_folder / 'rf_output/shap_rf_global_waterfall.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

print("\nFinding the customer with the highest probability of default...")

# Predict probabilities for the entire test set
probabilities = rf.predict_proba(X_test)

# Extract probabilities for Class 1 (Default)
default_probs = probabilities[:, 1]

# Find the index of the instance with the highest default probability
worst_customer_idx = np.argmax(default_probs)
max_prob = default_probs[worst_customer_idx]

# Get the specific row
instance = X_test.iloc[[worst_customer_idx]]

print("Generating Local Explanation object...")

local_explanation = explainer(instance)
local_exp_class1 = local_explanation[0, :, 1]

# Generate the Waterfall Plot
print("Rendering Waterfall plot...")
plt.figure(figsize=(10, 6))
shap.plots.waterfall(local_exp_class1, show=False)

# Save and display
output_path = script_folder / 'rf_output/shap_rf_local_waterfall.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")