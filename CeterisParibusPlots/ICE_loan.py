from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
from sklearn.inspection import PartialDependenceDisplay 
from sklearn.inspection import partial_dependence
import numpy as np
import dalex as dx
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

for col in X.columns:
    if col in categorical_columns or X[col].dtype == object or X[col].dtype == bool:
        le = sklearn.preprocessing.LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))


# Train Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, 
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

def predict_function(model, data):
    return rf.predict_proba(data)[:, 1]

print(f"Model Accuracy: {sklearn.metrics.accuracy_score(y_test, rf.predict(X_test))}")

# Create the Explainer
exp = dx.Explainer(rf, data=X_test, y=y_test,  predict_function=predict_function)

X_test_sampled = X_test.sample(n=100, random_state=42)
cp = exp.predict_profile(
    new_observation=X_test_sampled, 
    grid_points=40, 
    variables=['Income']
)

# Create plot
cp_plot = cp.plot(
    show=False,
    title='ICE Profiles for Income'
)

cp_plot.update_traces(
    opacity=0.3
)

cp_plot.update_layout(
    yaxis_title="Predicted Probability of Default",
    xaxis_title="Income (USD)",
    yaxis_range=[0, 1]
)

# Saving plot
output_path = script_folder / 'ice_income.html'
cp_plot.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

# Converted to float as required from sklearn
X_test_float = X_test_sampled.astype(float)


# ICE plot with sklearn

## Create ICE Plot
#display = PartialDependenceDisplay.from_estimator(
#    rf,
#    X_test_float,
#    features=['Income'],
#    target=['Default'],
#    kind='individual',
#    subsample=100,
#    random_state=42,
#    line_kw={"color": "blue", "alpha": 0.3}
#)
#
## Saving plot
#output_path = script_folder / 'ice_income.png'
#plt.savefig(output_path)
#print(f"Plot saved as '{output_path}'")



# Create Centered ICE Plot
display = PartialDependenceDisplay.from_estimator(
    rf,
    X_test_float,
    features=['Income'],
    target=['Default'],
    kind='individual',
    centered=True,
    subsample=100,
    random_state=42,
    line_kw={"color": "blue", "alpha": 0.3}
)

# Saving plot
output_path = script_folder / 'c-ice_income.png'
plt.savefig(output_path)
print(f"Plot saved as '{output_path}'")

# Derivative ICE plot

ice_results = partial_dependence(
    rf, 
    X_test_float, 
    features=['Income'], 
    kind='individual'
)

# Extract values
grid_values = ice_results["grid_values"][0] 
ice_values = ice_results["individual"][0]

# Compute derivative
d_ice_values = np.gradient(ice_values, grid_values, axis=1)

# Generate Plot
fig, ax = plt.subplots(figsize=(8, 5))

for i in range(d_ice_values.shape[0]):
    ax.plot(grid_values, d_ice_values[i], color='blue', alpha=0.3, linewidth=1)

# Adding
ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero derivative (No marginal effect)')

ax.set_title("Derivative ICE (d-ICE) per Income")
ax.set_xlabel("Income (USD)")
ax.set_ylabel("Predicted Probability Derivative")
ax.legend()
plt.tight_layout()

# Saving plot
output_path = script_folder / 'd-ice_income.png'
plt.savefig(output_path)
print(f"Plot saved as '{output_path}'")