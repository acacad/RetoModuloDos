import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a decision tree regressor model with limited complexity
regressor = DecisionTreeRegressor(**best_params, random_state=41)

# Train the model on the training set
regressor.fit(X_train, y_train)

# Make predictions on the validation and test sets
y_val_pred = regressor.predict(X_val)
y_test_pred = regressor.predict(X_test)

# Calculate R-squared for training, validation, and test sets
r2_train = r2_score(y_train, regressor.predict(X_train))
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"R-squared (train): {r2_train:.4f}")
print(f"R-squared (validation): {r2_val:.4f}")
print(f"R-squared (test): {r2_test:.4f}")

# Calculate Mean Squared Error for training, validation, and test sets
mse_train = mean_squared_error(y_train, regressor.predict(X_train))
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Store data to display the progression of training
train_sizes = np.arange(1, len(X_train) + 1)
train_errors = []
val_errors = []
test_errors = []

# Evaluate the model on growing subsets of the training data
for i in train_sizes:
    regressor.fit(X_train[:i], y_train[:i])
    train_errors.append(mean_squared_error(y_train[:i], regressor.predict(X_train[:i])))
    val_errors.append(mean_squared_error(y_val, regressor.predict(X_val)))
    test_errors.append(mean_squared_error(y_test, regressor.predict(X_test)))

# Plot the graph
plt.figure(figsize=(12, 6))

# Graph for the progression of error
plt.plot(train_sizes, val_errors, label='Validation Loss (MSE)', color='blue')
plt.plot(train_sizes, test_errors, label='Test Loss (MSE)', color='green')

plt.title('Progression of Loss (Mean Squared Error) for Decision Tree')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
