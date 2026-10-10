import effector
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
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

schema = effector.Schema(
    feature_names=feature_names,
    feature_types=[
        "ordinal", "ordinal", "ordinal", "ordinal", "ordinal", "ordinal",
        "ordinal", "ordinal", "ordinal", "nominal", "nominal", "nominal",
        "nominal", "nominal", "nominal", "nominal"
    ],
    category_names=category_names,
    target_name="Default",
)

# Convert dataframes in NumPy array to avoid compatability issues
X_test_np = X_test.values
y_test_np = y_test.values

predict = effector.adapters.classifier_proba(rf)

def predict_fn(x):
    df_x = pd.DataFrame(x, columns=feature_names)
    return predict(df_x)

print("Generating Effector report...")
report = effector.explain(
    X_test_np, 
    predict_fn, 
    y_test_np, 
    schema=schema, 
    nof_instances=100, 
    random_state=42
)

with open(script_folder / "report.html", "w", encoding="utf-8") as f:
    f.write(report.to_html())

print("Generating ALE plots...")
ale = effector.ALE(
    X_test_np, 
    predict_fn, 
    schema=schema, 
    nof_instances=100, 
    random_state=42
)

for i, col in enumerate(feature_names):
    ale.plot(i, show_plot=False)
    plt.savefig(script_folder / f"ale_plot_{col}.png", bbox_inches='tight', dpi=300)
    print("Plot saved as " / script_folder / f"ale_plot_{col}.png")