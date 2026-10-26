from pathlib import Path
import shap
import matplotlib.pyplot as plt
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

# Create the Explainer
print("\nInitializing SHAP Explainer...")
explainer = shap.PermutationExplainer(rf.predict_proba, X_test)

# For performance reasons during development, we calculate SHAP values on a sample of the test set.
X_test_sample = X_test.sample(n=500, random_state=42)

print("Calculating SHAP values...")
shap_values = explainer(X_test_sample)
shap_values_scatter = explainer(X_test[:500])[:, :, 1]

###################
### Local Plots ###
###################

local_instance = shap_values[0, :, 1]

# Generate the Bar Plot
print("Generating Bar plot...")

shap.plots.bar(local_instance, show=False)
plt.title("SHAP Local Bar Plot")

output_path = script_folder / 'rf_output/shap_rf_bar.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()

# Generate the Waterfall Plot
print("Generating Waterfall plot...")

shap.plots.waterfall(local_instance, show=False)
plt.title("SHAP Waterfall Plot")

output_path = script_folder / 'rf_output/shap_rf_waterfall.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()

# Generate the Force Plot
print("Generating Local Force plot...")

shap.plots.force(local_instance, matplotlib=True, show=False)
plt.title("SHAP Local Force Plot")

output_path = script_folder / 'rf_output/shap_rf_force.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()

####################
### Global Plots ###
####################

global_instances = shap_values[:, :, 1]

# Generate the Beeswarm Plot
print("Generating Beeswarm plot...")

shap.plots.beeswarm(global_instances, show=False)
plt.title("SHAP Beeswarm Plot")

output_path = script_folder / 'rf_output/shap_rf_beeswarm.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()

# Generate the Force Plot
print("Generating Global Force plot...")

force_plot_html = shap.plots.force(global_instances, show=False)

output_path = script_folder / 'rf_output/shap_rf_gforce.html'
shap.save_html(str(output_path), force_plot_html)
print(f"Plot saved as '{output_path}'")

# Generate the Global Bar (Importance) Plot
print("Generating Global Bar plot...")

shap.plots.bar(global_instances, show=False)
plt.title("SHAP Global Bar Plot")

output_path = script_folder / 'rf_output/shap_rf_gbar.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()

# Generate the Global Dependence Scatter Plot
print("Generating Global Dependence Scatter plot...")

shap.plots.scatter(shap_values_scatter[:, "Age"], show=False)
plt.title("SHAP Dependence Scatter Plot")

output_path = script_folder / 'rf_output/shap_rf_dependence.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()

# Generate the Decision Plot
print("Generating Decision plot...")

expected_value_class1 = 1
shap_values_array = shap_values.values[:, :, 1]
shap.decision_plot(expected_value_class1, shap_values_array, X_test_sample, show=False)
plt.title("SHAP Decision Plot")

output_path = script_folder / 'rf_output/shap_rf_decision.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"Plot saved as '{output_path}'")

plt.close()