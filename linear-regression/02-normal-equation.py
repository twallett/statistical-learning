#%%

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rsquared(y_true, y_pred):
    unexplained_var = ((y_true - y_pred)**2).sum()
    total_var = ((y_true - y_true.mean())**2).sum()
    return 1 - (unexplained_var / total_var)

data = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

w = np.zeros(X.shape[1])

w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred = X_test @ w

print("R Squared:", rsquared(y_test, y_pred).round(2))
print("Mean Squared Error:", mse(y_test, y_pred).round(2))

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = x_line @ w

plt.scatter(X_test, y_test, alpha=0.5, label='Test data')
plt.plot(x_line, y_line, color='red', label='Regression Line')

#%%