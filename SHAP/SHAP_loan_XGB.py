from pathlib import Path
import matplotlib.pylab as pl
import numpy as np
import xgboost
import shap
import joblib

np.random.seed(42)

# Loading Model-Dataset Bundle
script_folder = Path(__file__).parent
bundle_path = script_folder.parent / 'models' / 'loan_xgb.joblib'

print(f"Loading bundle from {bundle_path}...")
bundle = joblib.load(bundle_path)

gbtree = bundle['model']
X = bundle['X']
X_test = bundle['X_test']

# Classic feature attributions
output_path = script_folder / 'xgb_output/xgb_plot_importance.png'
xgboost.plot_importance(gbtree)
pl.title("xgboost.plot_importance(model)")
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

output_path = script_folder / 'xgb_output/xgb_plot_importance_cover.png'
xgboost.plot_importance(gbtree, importance_type="cover")
pl.title('xgboost.plot_importance(model, importance_type="cover")')
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

output_path = script_folder / 'xgb_output/xgb_plot_importance_gain.png'
xgboost.plot_importance(gbtree, importance_type="gain")
pl.title('xgboost.plot_importance(model, importance_type="gain")')
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# Explain predictions
print("Generating Explainer (This may take a while)...")
explainer = shap.TreeExplainer(gbtree)
shap_values = explainer.shap_values(X_test)

# Visualize a single prediction
output_path = script_folder / 'xgb_output/shap_xgb_force_plot_single.png'
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=True, show=False)
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# Visualize many (1000) predictions
output_path = script_folder / 'xgb_output/shap_xgb_force_plot_interactive.html'
interactive_plot = shap.force_plot(explainer.expected_value, shap_values[:1000, :], X_test.iloc[:1000, :])
shap.save_html(str(output_path), interactive_plot)
print(f"Plot saved as '{output_path}'")

# Bar chart of mean importance
output_path = script_folder / 'xgb_output/shap_xgb_summary_bar.png'
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, rng=42)
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")

# SHAP Summary Plot
output_path = script_folder / 'xgb_output/shap_xgb_summary_dots.png'
shap.summary_plot(shap_values, X_test, show=False, rng=42)
pl.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{output_path}'")