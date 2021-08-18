from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def get_top_k_indices(feature_importances, k):
    return np.sort(np.argpartition(np.array(feature_importances), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
        self.feature_indices_ = []

    def fit(self, X, y=None):
        self.feature_indices_ = get_top_k_indices(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]
