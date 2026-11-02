"""Additive Gaussian RBF regression with an X-only, fixed hidden layer.

Each original feature owns a block of Gaussian hidden units. Coalitions use
subsets of these blocks, so their unpenalized regression spaces are nested.
"""
from __future__ import annotations

import numpy as np


class AdditiveRBFNetwork:
    def __init__(self, center_quantiles=(0.2, 0.5, 0.8), width_multiplier=1.0):
        self.center_quantiles = tuple(center_quantiles)
        self.width_multiplier = float(width_multiplier)
        q = np.asarray(self.center_quantiles)
        if q.ndim != 1 or not len(q) or not np.all(np.isfinite(q)) or not np.all((q > 0) & (q < 1)):
            raise ValueError("Center quantiles must lie strictly inside (0, 1).")
        if len(np.unique(q)) != len(q) or not np.isfinite(self.width_multiplier) or self.width_multiplier <= 0:
            raise ValueError("Use distinct quantiles and a positive finite width multiplier.")

    @staticmethod
    def _array(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.size or not np.all(np.isfinite(X)):
            raise ValueError("X must be a nonempty finite matrix.")
        return X

    def fit_basis(self, X):
        """Freeze centers, widths, and training column means without seeing Y."""
        X = self._array(X)
        self.n_features_in_ = X.shape[1]
        self.centers_ = np.quantile(X, self.center_quantiles, axis=0).T
        scale = X.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError("Constant features are unsupported.")
        self.widths_ = self.width_multiplier * scale
        raw = self._raw(X)
        self.block_means_ = [b.mean(axis=0) for b in raw]
        return self

    def _raw(self, X):
        X = self._array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed.")
        return [np.exp(-0.5 * ((X[:, j, None] - self.centers_[j]) / self.widths_[j])**2)
                for j in range(self.n_features_in_)]

    def transform_blocks(self, X):
        return [b - mean for b, mean in zip(self._raw(X), self.block_means_)]

    def design_matrix(self, X, features=None):
        blocks = self.transform_blocks(X)
        features = tuple(range(self.n_features_in_)) if features is None else tuple(features)
        if len(set(features)) != len(features) or any(j < 0 or j >= self.n_features_in_ for j in features):
            raise ValueError("Invalid coalition.")
        return np.column_stack([np.ones(len(X)), *[blocks[j] for j in features]])

    def fit(self, X, y, features=None):
        """Refit output weights; retain an already frozen hidden layer."""
        if not hasattr(self, "centers_"):
            self.fit_basis(X)
        self.features_ = tuple(range(self.n_features_in_)) if features is None else tuple(features)
        a = self.design_matrix(X, self.features_)
        y = np.asarray(y, dtype=float)
        if y.shape != (a.shape[0],) or not np.all(np.isfinite(y)):
            raise ValueError("y must be a finite response vector matching X.")
        self.coef_ = np.linalg.lstsq(a, y, rcond=None)[0]
        return self

    def predict(self, X):
        return self.design_matrix(X, self.features_) @ self.coef_
