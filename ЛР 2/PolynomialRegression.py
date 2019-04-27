import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(object):

    def __init__(self, n_degree):
        self.n_degree = n_degree
        self.w = None
        self.transform_features = PolynomialFeatures(degree=n_degree)

    def fit(self, X, y):
        X_poly = self.transform_features.fit_transform(X)
        self.w = np.linalg.solve(np.dot(X_poly.T, X_poly), np.dot(X_poly.T, y))

    def predict(self, X):
        X_poly = self.transform_features.fit_transform(X)
        return np.dot(self.w.T, X_poly.T)
