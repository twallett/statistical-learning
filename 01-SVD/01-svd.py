#%%

from sklearn.datasets import load_diabetes
import numpy as np

x = load_diabetes()
X = x['data']

# eigval_v, V = np.linalg.eig(X.T @ X) # n x n
# eigval_u, U = np.linalg.eig(X @ X.T) # m x m - eigval_u is complex, eigval_h is not

# np.sqrt(np.real(eigval_u[:X.shape[1]])).sum() == np.sqrt(eigval_v[:X.shape[1]]).sum()

# the reason we do not compute Uh is because the signs are negative for < 0 

eigval_vh, Vh = np.linalg.eigh(X.T @ X) # n x n - eigh for hermetian or symmetric matrices

eigval_vh = eigval_vh[::-1]

Vh = Vh[:, ::-1]

r = X.shape[1]
sigma = np.sqrt(eigval_vh[:r])

Uh = (X @ Vh[:, :r]) / sigma
SIGMA = np.diag(sigma)

X_svd = Uh @ SIGMA @ Vh.T

np.allclose(np.abs(X), np.abs(X_svd))
# %%
