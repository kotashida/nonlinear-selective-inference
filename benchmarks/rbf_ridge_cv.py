"""X-only sampled additive bases, ridge coalition scores, and CV conditioning.

The score is improvement in min(RSS + n*lambda*||beta||^2), not ridge RSS.
CV is linear in y for fixed X and auxiliary random seeds. Its selection event
is therefore an intersection of quadratic inequalities, just like top-k.
"""
from math import ceil, comb

import run_rbf_shapley as core  # Cap BLAS threads before importing numpy.
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi


def basis(X, fraction, seed):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    indices = np.random.default_rng(seed).permutation(n)[:max(1, ceil(fraction*n))]
    centers = X[indices].T.copy()
    widths = []
    for j in range(p):
        distances = np.abs(X[:, j, None]-X[None, :, j])[np.triu_indices(n, 1)]
        width = float(np.median(distances))
        # Discrete/constant features: use the median positive distance, or 1.
        if width <= 0:
            positive = distances[distances > 0]
            width = float(np.median(positive)) if len(positive) else 1.
        widths.append(width)
    widths = np.asarray(widths)
    raw = [np.exp(-.5*((X[:, j, None]-centers[j])/widths[j])**2) for j in range(p)]
    means = [v.mean(axis=0) for v in raw]
    return dict(indices=indices, centers=centers, widths=widths, means=means)


def transform(X, fitted):
    return [np.exp(-.5*((X[:, j, None]-fitted['centers'][j])/fitted['widths'][j])**2)-fitted['means'][j]
            for j in range(X.shape[1])]


def cv_matrices(X, fraction, lambdas, folds, basis_seed, fold_seed):
    """Return C_l with CV_MSE_l(y)=y'C_l y; all preprocessing is fold-local."""
    n = len(X)
    order = np.random.default_rng(fold_seed).permutation(n)
    validation = np.array_split(order, folds)
    residuals = [np.eye(n) for _ in lambdas]
    fold_centers = []
    for fi, valid in enumerate(validation):
        train = np.setdiff1d(np.arange(n), valid)
        state = basis(X[train], fraction, np.random.SeedSequence([basis_seed, fi, 1]))
        fold_centers.append(train[state['indices']].tolist())
        Z = np.column_stack(transform(X[train], state))
        V = np.column_stack(transform(X[valid], state))
        U, singular, VT = np.linalg.svd(Z, full_matrices=False)
        for li, lam in enumerate(lambdas):
            weights = singular/(singular**2+len(train)*lam)
            prediction = np.ones((len(valid), len(train)))/len(train)+(V@VT.T*weights)@U.T
            residuals[li][np.ix_(valid, train)] = -prediction
    return [R.T@R/n for R in residuals], [v.tolist() for v in validation], fold_centers


class Geometry:
    def __init__(self, blocks, lam, tol=1e-10):
        self.p, self.tol = len(blocks), tol
        self.n = len(blocks[0])
        self.Q = core.orthobasis(np.column_stack(blocks), tol)
        self.r = self.Q.shape[1]
        reduced = [self.Q.T@b for b in blocks]
        grams = [v@v.T for v in reduced]
        smoothers = [np.zeros((self.r, self.r))]
        # H=K(K+n*lambda*I)^-1; the common unpenalized intercept cancels.
        for mask in range(1, 1 << self.p):
            K = sum(grams[j] for j in range(self.p) if mask & (1 << j))
            eigenvalues, U = np.linalg.eigh(K)
            eigenvalues = np.maximum(eigenvalues, 0.)
            H = (U*(eigenvalues/(eigenvalues+self.n*lam)))@U.T
            smoothers.append((H+H.T)/2)
        self.smoothers = smoothers
        self.selector = np.asarray([self.target('full_shapley', (1 << self.p)-1, j)[0]
                                    for j in range(self.p)])
        core.require(np.allclose(self.selector.sum(axis=0), smoothers[-1], atol=1e-8), 'Ridge Shapley efficiency')

    def select(self, y, k=2):
        scores = np.einsum('i,jik,k->j', y, self.selector, y)
        return tuple(sorted(int(j) for j in np.lexsort((np.arange(self.p), -scores))[:k]))

    def target(self, arm, M, j):
        universe = M if arm == 'selected_shapley' else (1 << self.p)-1
        size = universe.bit_count()
        A = np.zeros((self.r, self.r))
        for mask in range(1 << self.p):
            if mask & ~universe or mask & (1 << j):
                continue
            A += (self.smoothers[mask | (1 << j)]-self.smoothers[mask])/(size*comb(size-1, mask.bit_count()))
        A = (A+A.T)/2
        values, vectors = np.linalg.eigh(A)
        core.require(values.min(initial=0.) >= -1e-8, 'Non-PSD ridge score')
        B = vectors[:, values > self.tol*max(1., values.max(initial=0.))]
        core.require(np.allclose(B@B.T@A, A, atol=1e-8), 'Ridge target support')
        return A, B


def joint_intervals(g, z, u, selected, sigma, cv, chosen):
    """z,u are in the ORIGINAL response space, preserving CV's full dependence."""
    inequalities = []
    for j in selected:
        for other in range(g.p):
            if other not in selected:
                inequalities.append((g.Q@(g.selector[j]-g.selector[other])@g.Q.T, j < other))
    inequalities += [(C-cv[chosen], chosen < li) for li, C in enumerate(cv) if li != chosen]
    coefficients, boundaries = [], [0.]
    for D, tie in inequalities:
        a, b, c = float(sigma*sigma*u@D@u), float(2*sigma*u@D@z), float(z@D@z)
        # Clean only at floating-point roundoff, not at the rank tolerance.
        eps = 64*np.finfo(float).eps*max(1., np.linalg.norm(D, 2))
        a = 0. if abs(a) <= eps*sigma*sigma else a
        b = 0. if abs(b) <= eps*2*sigma*max(1., np.linalg.norm(z)) else b
        c = 0. if abs(c) <= eps*max(1., np.linalg.norm(z)**2) else c
        coefficients.append((a, b, c, tie))
        boundaries.extend(core.positive_roots(a, b, c))
    bounds = sorted(set(boundaries))+[np.inf]
    intervals = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mid = lo+max(1., lo) if np.isinf(hi) else lo+(hi-lo)/2
        if all((a*mid+b)*mid+c > 0 or ((a*mid+b)*mid+c == 0 and tie)
               for a, b, c, tie in coefficients):
            if intervals and intervals[-1][1] == lo:
                intervals[-1][1] = hi
            else:
                intervals.append([lo, hi])
    return intervals


def radial_test(g, y, B, selected, sigma, cv, chosen):
    rank = B.shape[1]
    if rank == 0:
        return dict(p_value=1., radius=0., rank=0, intervals=1, log_selection_mass=0.)
    W = g.Q@B
    projected = W@(W.T@y)
    radius = float(np.linalg.norm(projected)/sigma)
    core.require(radius > 0, 'Zero radius')
    u, z = projected/(sigma*radius), y-projected
    intervals = joint_intervals(g, z, u, selected, sigma, cv, chosen)
    core.require(any(lo-1e-7 <= radius <= hi+1e-7 for lo, hi in intervals), 'Observed radius lost')
    den = float(logsumexp([core.log_mass(lo, hi, rank) for lo, hi in intervals]))
    num = float(logsumexp([core.log_mass(max(lo, radius), hi, rank) for lo, hi in intervals]))
    core.require(np.isfinite(den), 'Nonfinite selection mass')
    p = float(np.exp(min(0., num-den)))
    core.require(np.isfinite(p), 'Nonfinite p-value')
    return dict(p_value=p, radius=radius, rank=rank, intervals=len(intervals), log_selection_mass=den,
                untruncated_reference_p=float(chi.sf(radius, rank)))
