from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import dice_ml
from dice_ml import Dice

np.random.seed(42)

# Loading Dataset
script_folder = Path(__file__).parent
dataset_path = script_folder.parent / 'data' / 'Loan_default.csv'
print("Loading Dataset...")
df = pd.read_csv(dataset_path)

# Defining target column
TARGET_COL = 'Default'

# Pre-Processing
# Removing rows with missing values
df = df.dropna()

# We must remove 'LoanID' (or any unique identifier) to avoid MemoryError.
COLS_TO_DROP = [TARGET_COL, 'LoanID']

X = df.drop(columns=COLS_TO_DROP)
y = df[TARGET_COL]

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y       
)

print(f"Train Size: {X_train.shape}")
print(f"Test Size: {X_test.shape}")

numerical_features=['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']
categorical_features = X_train.columns.difference(numerical_features)

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(
    steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Train Baseline

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])

print("Training Model...")
rf = clf.fit(X_train, y_train)

print(f"Model Accuracy: {accuracy_score(y_test, rf.predict(X_test))}")

# Initialize counterfactual generation methods

dice_data = dice_ml.Data(
    dataframe=df.drop(columns='LoanID'),
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

# Generate Counterfactuals
k = 3

dice_exp_random = exp_random.generate_counterfactuals(
    i, total_CFs=k, desired_class=0, verbose=False
)

dice_exp_genetic = exp_KD.generate_counterfactuals(
    i, total_CFs=k, desired_class=0, verbose=False
)

dice_exp_kd = exp_genetic.generate_counterfactuals(
    i, total_CFs=k, desired_class=0, verbose=False
)

# Save generated counterfactual examples to disk
# Function to add the original values (default ones) as the first row
def save_cf(dice_explanation, output_path):

    original_df = dice_explanation.cf_examples_list[0].test_instance_df
    
    counterfactual_df = dice_explanation.cf_examples_list[0].final_cfs_df
    
    result = pd.concat([original_df, counterfactual_df], ignore_index=True)
    
    result.to_csv(path_or_buf=output_path, index=False)
    print(f"CSV saved as '{output_path}'")

save_cf(dice_exp_random, script_folder / 'DiCE_random_counterfactuals.csv')
save_cf(dice_exp_genetic, script_folder / 'DiCE_kdtree_counterfactuals.csv')
save_cf(dice_exp_kd, script_folder / 'DiCE_genetic_counterfactuals.csv')


# Adding constraints...

# Assigning new weights
# Now generating explanations using the new feature weights, a higher feature weight means that the feature is harder to change than others
#feature_weights = {'Age': 10, 'Income': 5}
#dice_exp = dice_exp_random.generate_counterfactuals(
#    i, total_CFs=k, desired_class=0, feature_weights=feature_weights), verbose=False

# Fixing features
# DiCE allows imputting a list of features to vary, leaving unchanged the others

dice_fixed_exp_random = exp_random.generate_counterfactuals(
    i, total_CFs=k, desired_class=0, features_to_vary=['MonthsEmployed', 'InterestRate', 'LoanTerm', 'HasCoSigner'], verbose=False)

dice_fixed_exp_genetic = exp_random.generate_counterfactuals(
    i, total_CFs=k, desired_class=0, features_to_vary=['MonthsEmployed', 'InterestRate', 'LoanTerm', 'HasCoSigner'], verbose=False)

dice_fixed_exp_kd = exp_random.generate_counterfactuals(
    i, total_CFs=k, desired_class=0, features_to_vary=['MonthsEmployed', 'InterestRate', 'LoanTerm', 'HasCoSigner'], verbose=False)

save_cf(dice_fixed_exp_random, script_folder / 'DiCE_fixed_random_counterfactuals.csv')
save_cf(dice_fixed_exp_genetic, script_folder / 'DiCE_fixed_kdtree_counterfactuals.csv')
save_cf(dice_fixed_exp_kd, script_folder / 'DiCE_fixed_genetic_counterfactuals.csv')