"""Fixed-design linear RSS Shapley scores, with reusable quadratic forms."""

from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np


def _basis(design):
    if design.shape[1] == 0:
        return np.empty((design.shape[0], 0))
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(float).eps * s[0]
    return u[:, s > tolerance]


class LinearShapleyStatistic:
    """Precompute T_j(y) = y.T A_j y for a fixed finite design.

    All coalitions include an intercept by default; without an intercept the
    empty model has fitted value zero. Monte Carlo averages uniform random
    permutations (their predecessor sets have exactly the Shapley weights).
    Randomness is fixed at construction and reused for every response.
    Quadratic forms are stored in the full design's orthonormal coordinates,
    reducing storage from O(p*n*n) to O(p*r*r), r <= min(n, p).
    """

    def __init__(self, X, *, mode="auto", exact_max_features=10,
                 n_permutations=1000, random_state=42, fit_intercept=True):
        X = np.array(X, dtype=float, copy=True)
        if X.ndim != 2 or min(X.shape) < 1 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a nonempty finite two-dimensional array.")
        if mode not in {"auto", "exact", "monte_carlo"}:
            raise ValueError("mode must be 'auto', 'exact', or 'monte_carlo'.")
        for name, value in (("exact_max_features", exact_max_features),
                            ("n_permutations", n_permutations)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)) or random_state < 0:
            raise ValueError("random_state must be a fixed nonnegative integer.")
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean.")
        self.X = X
        self.X.setflags(write=False)
        self.fit_intercept = bool(fit_intercept)
        p = X.shape[1]
        self.mode = ("exact" if p <= exact_max_features else "monte_carlo") if mode == "auto" else mode
        design = X - X.mean(axis=0) if fit_intercept else X.copy()
        norms = np.linalg.norm(design, axis=0)
        design = np.divide(design, norms, out=np.zeros_like(design), where=norms > 0)
        self.basis = _basis(design)
        coordinates = self.basis.T @ design
        r = self.basis.shape[1]
        self.reduced_matrices = np.zeros((p, r, r))

        @lru_cache(maxsize=256)
        def coalition_basis(mask):
            columns = [j for j in range(p) if mask & (1 << j)]
            return _basis(coordinates[:, columns]) if r else np.empty((0, 0))

        def add(j, mask, weight):
            q = coalition_basis(mask)
            v = coordinates[:, j] - q @ (q.T @ coordinates[:, j])
            # Reorthogonalization avoids spurious contributions for aliases.
            v -= q @ (q.T @ v)
            norm = np.linalg.norm(v)
            if norm > max(design.shape) * np.finfo(float).eps * 8:
                v /= norm
                self.reduced_matrices[j] += weight * np.outer(v, v)

        if self.mode == "exact":
            for mask in range(1 << p):
                size = mask.bit_count()
                if size == p:
                    continue
                weight = 1.0 / (p * comb(p - 1, size))
                for j in range(p):
                    if not mask & (1 << j):
                        add(j, mask, weight)
        else:
            rng = np.random.default_rng(random_state)
            for _ in range(n_permutations):
                mask = 0
                for j in rng.permutation(p):
                    j = int(j)
                    add(j, mask, 1.0 / n_permutations)
                    mask |= 1 << j
        self.basis.setflags(write=False)
        self.reduced_matrices.setflags(write=False)
        self._support_cache = {}

    @property
    def matrices(self):
        """Materialize A_j in response coordinates (O(p*n*n) storage)."""
        return np.einsum("na,jab,mb->jnm", self.basis, self.reduced_matrices, self.basis)

    def scores(self, response):
        response = np.asarray(response, dtype=float)
        if response.shape != (self.X.shape[0],) or not np.all(np.isfinite(response)):
            raise ValueError("response must be a finite vector matching X rows.")
        z = self.basis.T @ response
        return np.maximum(0.0, np.einsum("a,jab,b->j", z, self.reduced_matrices, z))

    def support_basis(self, feature):
        """Orthonormal range of A_j for the null A_j mu = 0."""
        if feature not in self._support_cache:
            eigenvalues, vectors = np.linalg.eigh(self.reduced_matrices[feature])
            tolerance = (max(self.X.shape) * np.finfo(float).eps *
                         max(1.0, float(eigenvalues.max(initial=0.0))))
            self._support_cache[feature] = self.basis @ vectors[:, eigenvalues > tolerance]
        return self._support_cache[feature]
