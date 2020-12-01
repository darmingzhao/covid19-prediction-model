import numpy as np
from numpy.linalg import solve, lstsq
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self, X, y, lam):
        D = X.shape[1]
        ### Using L2-Regularization
        self.w = solve(X.T@X + lam*np.eye(D), X.T@y)

    def predict(self, X):
        return X@self.w
