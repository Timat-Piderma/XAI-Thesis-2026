from __future__ import print_function
from pathlib import Path
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import joblib

### AGGIUNTO: Ignora i warning di scikit-learn sui nomi delle feature mancanti in LIME ###
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
##########################################################################################

np.random.seed(42)

# Loading Model-Dataset Bundle
script_folder = Path(__file__).parent
bundle_path = script_folder.parent / 'models' / 'loan_rf.joblib'

print(f"Loading bundle from {bundle_path}...")
bundle = joblib.load(bundle_path)

rf = bundle['model']
X_test = bundle['X_test']
y_test = bundle['y_test']
feature_names = bundle['feature_names'].tolist()
categorical_names_bundle = bundle['categorical_names']
categorical_cols = bundle['categorical_cols']

print("Applying Label Encoding for LIME...")

# Create features dictionary
categorical_features = []
categorical_names = {}

for i, col_name in enumerate(categorical_cols):
    col_idx = X_test.columns.get_loc(col_name)
    categorical_features.append(col_idx)
    categorical_names[col_idx] = categorical_names_bundle[i]

# Convert dataframes in NumPy array to avoid compatability issues
X_test_np = X_test.values
y_test_np = y_test.values

def predict_proba_fn(x):
    df_x = pd.DataFrame(x, columns=feature_names)
    return rf.predict_proba(df_x)

# Create the Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_test_np, 
    feature_names=feature_names, 
    class_names=['No Default', 'Default'], 
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    discretize_continuous=True, 
    mode='classification'
)

# Explaining a non Default instances
prd = rf.predict(X_test)
prd_def = np.where(prd == 0)[0]
i = prd_def[0]

exp = explainer.explain_instance(
    X_test_np[i],
    predict_proba_fn,  
    top_labels=1
)

# Save plot
output_path = script_folder / 'rf_output/lime_rf_single_explanation_nondefault.html'
exp.save_to_file(file_path=output_path, show_table=True, show_all=False)
print(f"Plot saved as '{output_path}'")

# Explaining a default instances
prd_ndef = np.where(prd == 1)[0]
i = prd_ndef[0]

exp = explainer.explain_instance(
    X_test_np[i],
    predict_proba_fn,  
    top_labels=1
)

# Save plot
output_path = script_folder / 'rf_output/lime_rf_single_explanation_default.html'
exp.save_to_file(file_path=output_path, show_table=True, show_all=False)
print(f"Plot saved as '{output_path}'")