#%%

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

data = make_classification(n_samples=1000, n_features=10, n_classes=2)

X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
w = np.zeros((X.shape[1], 1))

def sigmoid(line):
    return (1 / (1 + np.exp(-line)))

def binary_cross_entropy(actual, pred):
    return -1 * (actual * np.log(pred + 1e-10) + (1- actual) * np.log((1-pred + 1e-10))).mean()

for epoch in range(500):
    y_pred = sigmoid(X_train @ w)
    loss = binary_cross_entropy(y_train, y_pred)
    grad = (y_train - y_pred)
    w += 1e-03 * (X_train.T @ grad)
    print(loss)
    
predictions = (sigmoid(X_test @ w) > 0.5).astype(int)
    
def accuracy(actual, pred):
    tp_tn = sum([1 if a_i == p_i else 0 for a_i, p_i in zip(actual, pred)])
    return tp_tn / len(actual)

acc = accuracy(y_test, predictions)
print(acc)

# %%
