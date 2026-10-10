from pathlib import Path
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

# Get one observation from test values
obs = X_test.sample(n=1, random_state=42)

cp = exp.predict_profile(
    new_observation=obs, 
    variables=['Income']
)

# Create plot
plt = cp.plot(
    show=False,
    title='Ceteris Paribus for Income'
)

plt.update_traces(
    line=dict(color='red', width=3, shape='spline', smoothing=1.3), 
    marker=dict(size=10, color='darkred', symbol='circle', line=dict(color='black', width=1))
)

plt.update_layout(
    yaxis_title="Predicted Probability of Default",
    xaxis_title="Income (USD)",
    yaxis_range=[0, 1]
)

# Saving plot
output_path = script_folder / 'cp_income.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")