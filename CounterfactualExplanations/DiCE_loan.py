from pathlib import Path
import pandas as pd
import numpy as np
import dice_ml
from dice_ml import Dice
import joblib

np.random.seed(42)

# Loading Model-Dataset Bundle
script_folder = Path(__file__).parent
bundle_path = script_folder.parent / 'models' / 'loan_rf.joblib'

print(f"Loading bundle from {bundle_path}...")
bundle = joblib.load(bundle_path)

rf = bundle['model']
X = bundle['X'] 
y = bundle['y'] 
X_test = bundle['X_test']
y_test = bundle['y_test']
X_train = bundle['X_train']
y_train = bundle['y_train']

# DiCE require a dataframe containing all features
X_encoded = pd.concat([X_train, X_test])
y_encoded = pd.concat([y_train, y_test])

df = X_encoded.copy()
df['Default'] = y_encoded

# Initialize counterfactual generation methods

dice_data = dice_ml.Data(
    dataframe=df,
    continuous_features=['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio'],
    outcome_name='Default'
)

dice_model = dice_ml.Model(
    model=rf,
    backend="sklearn"
)

print("Initialize Explainers...")
exp_random = Dice(dice_data, dice_model, method="random")
exp_genetic = Dice(dice_data, dice_model, method="genetic")
exp_KD = Dice(dice_data, dice_model, method="kdtree")

# Extract Random Default Data Entry
prd = rf.predict(X_test)
default_indexes = np.where(prd == 1)[0]

i = X_test.iloc[[np.random.choice(default_indexes)]]

# Save generated counterfactual examples to disk
# Function to add the original values (default ones) as the first row
def save_cf(dice_explanation, output_path):

    original_df = dice_explanation.cf_examples_list[0].test_instance_df
    
    counterfactual_df = dice_explanation.cf_examples_list[0].final_cfs_df
    
    result = pd.concat([original_df, counterfactual_df], ignore_index=True)
    
    result.to_csv(path_or_buf=output_path, index=False)
    print(f"CSV saved as '{output_path}'")

# Generate Counterfactuals
k = 3

try:
    dice_exp_random = exp_random.generate_counterfactuals(
        i, total_CFs=k, desired_class=0, verbose=False)
    save_cf(dice_exp_random, script_folder / 'DiCE_random_counterfactuals.csv')
except:
    print("Exception raised for dice_exp_random...")

try:
    dice_exp_genetic = exp_genetic.generate_counterfactuals(
        i, total_CFs=k, desired_class=0, verbose=False)
    save_cf(dice_exp_genetic, script_folder / 'DiCE_genetic_counterfactuals.csv')
except:
    print("Exception raised for dice_exp_genetic...")

try:
    dice_exp_kd = exp_KD.generate_counterfactuals(
        i, total_CFs=k, desired_class=0, verbose=False)
    save_cf(dice_exp_kd, script_folder / 'DiCE_kd_counterfactuals.csv')
except:
    print("Exception raised for dice_exp_kd...")

# Adding constraints...

# Assigning new weights
# Now generating explanations using the new feature weights, a higher feature weight means that the feature is harder to change than others
#feature_weights = {'Age': 10, 'Income': 5}
#dice_weighted_exp = dice_exp_random.generate_counterfactuals(
#    i, total_CFs=k, desired_class=0, feature_weights=feature_weights, verbose=False)

# Fixing features
# DiCE allows imputting a list of features to vary, leaving unchanged the others
try:
    dice_fixed_exp_random = exp_random.generate_counterfactuals(
        i, total_CFs=k, desired_class=0, features_to_vary=['LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm', 'HasCoSigner'], verbose=False)
    save_cf(dice_fixed_exp_random, script_folder / 'DiCE_fixed_random_counterfactuals.csv')
except:
    print("Exception raised for dice_fixed_exp_random...")

try:
    dice_fixed_exp_genetic = exp_genetic.generate_counterfactuals(
        i, total_CFs=k, desired_class=0, features_to_vary=['LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm', 'HasCoSigner'], verbose=False)
    save_cf(dice_fixed_exp_genetic, script_folder / 'DiCE_fixed_genetic_counterfactuals.csv')
except:
    print("Exception raised for dice_fixed_exp_genetic...")

try:
    dice_fixed_exp_kd = exp_KD.generate_counterfactuals(
        i, total_CFs=k, desired_class=0, features_to_vary=['LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm', 'HasCoSigner'], verbose=False)
    save_cf(dice_fixed_exp_kd, script_folder / 'DiCE_fixed_kd_counterfactuals.csv')
except:
    print("Exception raised for dice_fixed_exp_kd...")



