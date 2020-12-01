import numpy as np
import utils

from linear_model import LeastSquares

# Autoregressive Linear Model
class LinearModelAutoregressive:
    
    def __init__(self, K):
        self.K = K

    def fit(self, D, lam):
        X, y, x = self.__arBasis(D)

        model = LeastSquares()
        '''
        y = [[d_K],[d_{K+1}], ...[d_T], [c_K], [c_{K+1}], ...[c_T]]
        '''
        model.fit(X, y, lam)

        self.w = model.w
        self.x = x
        self.T, self.F = D.shape

    def predict(self, start, end):
        w = self.w
        x = self.x.copy()
        K = self.K

        y_pred = np.zeros((end - start, self.F))
        # We predict for each x
        for i in range(start, end):
            
            y = x@w
            y_pred[i - start] = y

            # Shift over to the left by 1 day for each feature
            # Then insert the latest results per feature
            length = K-1
            for j in range(self.F):
                x[1+(length)*j:1+(length)*(j+1)] = x[2+(length)*j:2+(length)*(j+1)]
                x[:, (length)*(j+1)] = y[:, j]
        
        return y_pred

    # Adds all ones in the first column
    def __biasBasis(self, X):
        n, d = X.shape
        Z = np.ones((n, d + 1))
        Z[:,1:] = X
        return Z
    
    # Outputs X, y, x in Autoregression basis
    def __arBasis(self, D):
        K = self.K

        # T is number of examples, F is number of features
        T, F = D.shape
        D = D.T


        X = np.ones((T - K + 1, F*(K-1)))
        y = np.zeros((T - K + 1, F))

        y = D[:, K-1:T].T

        # There are F features
        # The first K-1 columns in X correspond to feature 1
        #     first column being values of feature 1 on days 1...T-K+1
        #     second column being values of feature 1 on days 2...T-K+2
        #     K-1th column being values of feature 1 on days K-1...T-1
        # The second K-1 columns in X correspond to feature 2, follow same pattern
        # ...
        # The last K-1 features in X correspond to feature F
        for i in range(0, T - K + 1):
            for j in range(0, F):
                X[i, j*(K-1):(j+1)*(K-1)] = D[j, i:i+K-1]
        
        x = np.zeros((1, F*(K-1)))
        for j in range(0, F):
            x[:, j*(K-1):(j+1)*(K-1)] = D[j, T-K+1:]     

        # Bias basis adds a column of ones to the very left in X and x
        X = self.__biasBasis(X)
        x = self.__biasBasis(x)

        return (X.astype(float), y.astype(float), x)
