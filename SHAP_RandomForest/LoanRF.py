import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Loading Dataset
print("Loading dataset...")
df = pd.read_csv('data/Loan_default.csv')

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

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y       
)

print(f"Train Size: {X_train.shape}")
print(f"Test Size: {X_test.shape}")

# Train Baseline
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Valutation
print("\nGenerating Predictions...")
y_pred = rf_model.predict(X_test)

print("\n--- RESULTS ---")
print(f"Global Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

import shap
import matplotlib.pyplot as plt

# Shap Global Explanation
print("\nInitializing SHAP TreeExplainer...")

explainer = shap.TreeExplainer(rf_model)

# For performance reasons during development, we calculate SHAP values on a sample of the test set.
X_test_sample = X_test.sample(n=500, random_state=42)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test_sample)

# For Random Forest classification, shap_values is a list of arrays (one for each class).
# In a binary classification problem (Default vs No Default), we usually care about the positive class.
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

# Save the plot 
plt.tight_layout()
plt.savefig('shap_global_summary.png', dpi=300)
print("Plot saved as 'shap_global_summary.png'.")
plt.show()

import numpy as np

print("\nFinding the customer with the highest probability of default...")

# Predict probabilities for the entire test set
probabilities = rf_model.predict_proba(X_test)

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
plt.tight_layout()
plt.savefig('shap_local_waterfall.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'shap_local_waterfall.png'.")
plt.show()