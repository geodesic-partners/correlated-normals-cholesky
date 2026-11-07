"""Correlated normal random vectors from a correlation or covariance matrix, by Cholesky
factorisation, in pure Python. Refuses matrices that are not symmetric positive definite."""
import math
import random


def cholesky(a, tol=1e-12):
    """Return lower-triangular L with L L^T = a. Raises ValueError if a is not symmetric positive definite."""
    n = len(a)
    for i in range(n):
        if len(a[i]) != n:
            raise ValueError("matrix must be square")
        for j in range(i):
            if abs(a[i][j] - a[j][i]) > tol * max(1.0, abs(a[i][j])):
                raise ValueError("matrix must be symmetric")
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= tol:
                    raise ValueError("matrix is not positive definite (pivot %.3g at row %d)" % (s, i))
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def correlated_normals(cov, n_samples, mean=None, rng=random):
    """Draw n_samples vectors x = mean + L z with z independent standard normals, so Cov(x) = cov."""
    L = cholesky(cov)
    d = len(cov)
    mu = mean or [0.0] * d
    out = []
    for _ in range(n_samples):
        z = [rng.gauss(0.0, 1.0) for _ in range(d)]
        out.append([mu[i] + sum(L[i][k] * z[k] for k in range(i + 1)) for i in range(d)])
    return out


def sample_cov(xs):
    n, d = len(xs), len(xs[0])
    m = [sum(x[i] for x in xs) / n for i in range(d)]
    return [[sum((x[i] - m[i]) * (x[j] - m[j]) for x in xs) / (n - 1) for j in range(d)] for i in range(d)]
