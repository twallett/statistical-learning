#%%

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

data = make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)

X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
w = np.zeros(X.shape[1]).reshape(-1,1)

loss_history = []
w_history = np.zeros((500,X.shape[1]))

def sign(w):
    w[w>0] = 1
    w[w<0] = -1
    return w

penalty = 0.4

for epoch in range(500):
    y_pred_train = X_train @ w
    loss = (1/len(y_train)) * np.linalg.norm(y_train - y_pred_train) ** 2 + (penalty * np.linalg.norm(w, ord= 1))
    loss_history.append(loss)
    w_history[epoch] += w.reshape(X.shape[1])
    gradient = X_train.T @ (y_train - y_pred_train) - (penalty * sign(w)) 
    w += 1e-3 * gradient 
    print(w)

def rsquared(y_true, y_pred):
    unexplained_var = ((y_true - y_pred)**2).sum()
    total_var = ((y_true - y_true.mean())**2).sum()
    return 1 - (unexplained_var / total_var)

y_pred = X_test @ w
print("R Squared:", rsquared(y_test, y_pred).round(2))

plt.plot(w_history)

# %%
