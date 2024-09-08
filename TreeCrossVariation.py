import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

best_params = {
    'criterion': 'squared_error',
    'splitter': 'best',
    'max_depth': 6,
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0
}

# Load the dataset
columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
df = pd.read_csv('abalone.data', names=columns)

# Preprocessing: Convert the categorical variable "Sex" to numeric
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})

# Separate features and target
X = df.drop('Rings', axis=1)  # All columns except "Rings"
y = df['Rings']  # The "Rings" column is the target

# Create a decision tree regressor model with limited complexity
regressor = DecisionTreeRegressor(**best_params, random_state=41)

# Set up KFold cross-validation with k=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate MSE for each fold
mse_scores = -cross_val_score(regressor, X, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(regressor, X, y, cv=kf, scoring='r2')

# Calculate the variance of the scores
mse_variance = np.var(mse_scores)
r2_variance = np.var(r2_scores)

# Print the results
print(f"MSE scores for each fold: {mse_scores}")
print(f"Mean MSE: {mse_scores.mean():.4f}")
print(f"Variance of MSE: {mse_variance:.4f}")
print(f"R-squared scores for each fold: {r2_scores}")
print(f"Mean R-squared: {r2_scores.mean():.4f}")
print(f"Variance of R-squared: {r2_variance:.4f}")
