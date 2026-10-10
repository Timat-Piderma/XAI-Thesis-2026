from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay 
from sklearn.inspection import partial_dependence
import numpy as np
import dalex as dx
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

def predict_function(model, data):
    return model.predict_proba(data)[:, 1]

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