import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#VERSION FINALE DU PREMIER MODEL

# 1. Load the data
data = pd.read_csv('abalone.data', header=None)
data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

# Normalize "Sex" column
mean_female_rings = 11.129304
mean_male_rings = 10.705497
mean_infant_rings = 7.890462

normalized_female_sex = 1
normalized_male_sex = mean_male_rings / mean_female_rings
normalized_infant_sex = mean_infant_rings / mean_female_rings

data['Sex'] = data['Sex'].map({'M': normalized_male_sex, 'F': normalized_female_sex, 'I': normalized_infant_sex})

# Drop unused columns
data = data.drop(["Height"], axis=1)

# Shuffle data
data = shuffle(data, random_state=0)

# Split data into training, validation, and test sets
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
test_size = len(data) - train_size - val_size

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize weights
m_train, n = X_train.shape
theta = np.zeros(n)

# Define the gradient descent and R-squared functions
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error**2)
    return cost

def gradient_descent(X_train, y_train, X_val, y_val, theta, learning_rate, epochs):
    m_train = len(y_train)
    m_val = len(y_val)
    cost_history_train = np.zeros(epochs)
    cost_history_val = np.zeros(epochs)
    accuracy_train = np.zeros(epochs)
    accuracy_val = np.zeros(epochs)

    for i in range(epochs):
        # Training set
        predictions_train = X_train.dot(theta)
        error_train = predictions_train - y_train
        theta -= (learning_rate / m_train) * (X_train.T.dot(error_train))
        cost_history_train[i] = compute_cost(X_train, y_train, theta)

        # Validation set
        predictions_val = X_val.dot(theta)
        error_val = predictions_val - y_val
        cost_history_val[i] = compute_cost(X_val, y_val, theta)

        # Calculate accuracy based on a tolerance threshold
        tolerance = 1  # You can adjust this threshold as needed
        accuracy_train[i] = np.mean(np.abs(error_train) <= tolerance)
        accuracy_val[i] = np.mean(np.abs(error_val) <= tolerance)

    return theta, cost_history_train, cost_history_val, accuracy_train, accuracy_val

def r_squared(X, y, theta):
    predictions = X.dot(theta)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    return 1 - (ss_residual / ss_total)

# Parameters for gradient descent
learning_rate = 0.1
epochs = 120

# Train the model
theta, cost_history_train, cost_history_val, accuracy_train, accuracy_val = gradient_descent(X_train, y_train, X_val, y_val, theta, learning_rate, epochs)

# Evaluate the model
r2_val = r_squared(X_val, y_val, theta)
r2_test = r_squared(X_test, y_test, theta)

# Calculate the percentage of correct predictions for the validation set
predictions_val = X_val.dot(theta)
error_val = np.abs(predictions_val - y_val)
percentage_correct_val = 100 - (np.mean(error_val) / np.mean(y_val) * 100)

# Print results
print(f"Validation R-squared: {r2_val:.4f}")
print(f"Test R-squared: {r2_test:.4f}")
print(f"Validation Correct Predictions Percentage: {percentage_correct_val:.2f}%")

# Print coefficients
print("\nCoefficients (theta values):")
for i, coeff in enumerate(theta):
    if i == 0:
        print(f"Bias term: {coeff}")
    else:
        print(f"Coefficient for feature {data.columns[i-1]}: {coeff}")

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), accuracy_train, 'r-', label='Training accuracy')
plt.plot(range(epochs), accuracy_val, 'b-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(range(epochs), cost_history_train, 'r-', label='Training loss')
plt.plot(range(epochs), cost_history_val, 'b-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
