#%%

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

data = make_regression(n_samples=1000, n_features=10, n_informative=3, noise=20, random_state=42)

X = data[0]
y = data[1]

X_train_scaled, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scaled)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
w = np.zeros(X.shape[1]).reshape(-1,1)

loss_history = []
w_history = np.zeros((1000,X.shape[1]))

def soft_threshold(z, penalty):
    if z > penalty:
        z -= penalty
    elif abs(z) <= penalty:
        z = 0
    else:
        z += penalty
    return z

penalty = 1 * len(X_train_scaled)

for epoch in range(1000):
    y_pred_train = X_train_scaled @ w
    loss = (1/len(y_train)) * np.linalg.norm(y_train - y_pred_train) ** 2 + (penalty * np.linalg.norm(w, ord= 1))
    loss_history.append(loss)
    for indx, j in enumerate(w):
        residual_j = y_train - (X_train_scaled @ w) + (X_train_scaled[:,indx].reshape(-1,1) @ w[indx]).reshape(-1,1)
        z_j = (X_train_scaled[:, indx].T @ residual_j).item()
        w[indx] = (soft_threshold(z_j, penalty)/np.linalg.norm(X_train_scaled[:, indx])**2) 
    w_history[epoch] = w.reshape(-1)

def rsquared(y_true, y_pred):
    unexplained_var = ((y_true - y_pred)**2).sum()
    total_var = ((y_true - y_true.mean())**2).sum()
    return 1 - (unexplained_var / total_var)


print("Final weights:", w.flatten())
y_pred = X_test @ w
print("R Squared:", rsquared(y_test, y_pred).round(2))

plt.plot(w_history)

# %%
