from __future__ import print_function
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

pdp = exp.model_profile(
    variables=['Age','Income','LoanAmount','CreditScore'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_1.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['MonthsEmployed','NumCreditLines','InterestRate','LoanTerm'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_2.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['DTIRatio','Education','EmploymentType','MaritalStatus'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_3.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['HasMortgage','HasDependents','LoanPurpose','HasCoSigner'],
    N=100)

plt=pdp.plot(geom='profiles', show=False)

output_path = script_folder / 'dalex_ice_pdp_4.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")

pdp = exp.model_profile(
    variables=['Income'],
    N=100)

plt=pdp.plot(geom='profiles', show=False, title='PDP Profile for Income')

plt.update_layout(
    yaxis_title="Predicted Probability of Default",
    xaxis_title="Income (USD)",
    yaxis_range=[0, 1]
)

output_path = script_folder / 'dalex_ice_pdp_income.html'
plt.write_html(str(output_path))
print(f"Plot saved as '{output_path}'")