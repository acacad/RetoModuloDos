import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#PREMIER MODEL AVEC CROSS VALIDATION


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

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term
X = np.c_[np.ones(X.shape[0]), X]

# Define the gradient descent and R-squared functions
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error**2)
    return cost

def gradient_descent(X, y, theta, learning_rate, epochs):
    m = len(y)
    cost_history = np.zeros(epochs)
    for i in range(epochs):
        predictions = X.dot(theta)
        error = predictions - y
        theta -= (learning_rate / m) * (X.T.dot(error))
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

def r_squared(X, y, theta):
    predictions = X.dot(theta)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    return 1 - (ss_residual / ss_total)

# Parameters for gradient descent
learning_rate = 0.01
epochs = 10000

# Cross-validation with k-folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=0)

r2_scores = []
coefficients_list = []
cost_histories = []

# Loop over the k folds
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize weights
    m_train, n = X_train.shape
    theta = np.zeros(n)

    # Train the model
    theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, epochs)

    # Evaluate the model
    r2_test = r_squared(X_test, y_test, theta)

    # Store results
    r2_scores.append(r2_test)
    coefficients_list.append(theta)
    cost_histories.append(cost_history)

    # Print R-squared for the current fold
    print(f"R-squared for fold {fold}: {r2_test:.4f}")

# Calculate average and variance of R-squared
avg_r2 = np.mean(r2_scores)
variance_r2 = np.var(r2_scores)

print(f"\nAverage R-squared over {k} folds: {avg_r2:.4f}")
print(f"Variance of R-squared over {k} folds: {variance_r2:.4f}")

# Display coefficients from the last fold
print("\nCoefficients from the last fold (theta values):")
for i, coeff in enumerate(coefficients_list[-1]):
    if i == 0:
        print(f"Bias term: {coeff}")
    else:
        print(f"Coefficient for feature {data.columns[i-1]}: {coeff}")

# Plot cost function over epochs for the last fold
plt.plot(range(epochs), cost_histories[-1], 'b-')
plt.title('Cost over Epochs (Last Fold)')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)
plt.show()
