from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from anchor import anchor_tabular
import joblib

np.random.seed(42)

# Loading Model-Dataset Bundle
script_folder = Path(__file__).parent
bundle_path = script_folder.parent / 'models' / 'loan_rf.joblib'

print(f"Loading bundle from {bundle_path}...")
bundle = joblib.load(bundle_path)

rf = bundle['model']
X_test = bundle['X_test']
feature_names = bundle['feature_names']
categorical_names = bundle['categorical_names']
categorical_cols = bundle['categorical_cols']

predict_fn = lambda x: rf.predict(x)

categorical_names_dict = {}

for i, col_name in enumerate(categorical_cols):
    col_idx = X_test.columns.get_loc(col_name)
    categorical_names_dict[col_idx] = categorical_names[i]

# Convert dataframes in NumPy array to avoid compatability issues
X_test_np = X_test.values

# Create Explainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names= ['No Default', 'Default'],
    feature_names= feature_names,
    train_data= X_test_np,
    categorical_names= categorical_names_dict
)

idx = np.random.randint(0, X_test_np.shape[0])

# Generate explanation
exp = explainer.explain_instance(X_test_np[idx], predict_fn, threshold=0.95)

# Save output
with open(script_folder / "output.txt", "w", encoding="utf-8") as f:
    print('Prediction: ', explainer.class_names[predict_fn(X_test_np[idx].reshape(1, -1))[0]], file=f)
    print('Anchor: %s' % (' AND '.join(exp.names())), file=f)
    print('Precision: %.2f' % exp.precision(), file=f)
    print('Coverage: %.2f' % exp.coverage(), file=f)

print("Output saved in " / script_folder / "output.txt")