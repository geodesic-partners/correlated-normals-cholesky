# Correlated normal random numbers by Cholesky factorisation, in pure Python

If you price anything on more than one asset by Monte Carlo, you'll need normal draws with a given correlation, and "how would you generate them?" is a question I'd expect in a quant developer or quant researcher interview. The standard answer is Cholesky. You factor the covariance matrix as L L^T with L lower triangular, draw independent standard normals z, and x = mean + L z then has exactly the covariance you asked for.

```python
from corr import cholesky, correlated_normals

cholesky([[1.0, 0.6], [0.6, 1.0]])          # [[1.0, 0.0], [0.6, 0.8]]
xs = correlated_normals([[1.0, 0.7], [0.7, 2.0]], 10_000, mean=[0.0, 1.0])
```

For two assets with correlation rho, L works out to [[1, 0], [rho, sqrt(1 - rho^2)]], which is the formula many people remember without knowing where it comes from. The part I think interviewers care about most is what happens with a matrix that can't be a correlation matrix. Pairwise correlations of 0.9, 0.9 and -0.9 look harmless one at a time, but they can't all hold at once (the matrix has an eigenvalue of -0.8), and Cholesky finds that out by hitting a non-positive pivot. This code raises a `ValueError` saying which row failed, so you don't get a quiet NaN halfway through a simulation.

## Tests

`python3 -m unittest test_corr` checks that L L^T gives back a 3 by 3 covariance matrix to 12 decimal places, the two-asset closed form, that the impossible correlation matrix and a non-symmetric matrix are both refused, and that 100,000 draws on a fixed seed reproduce the means and the covariance matrix to within 0.03. You'll only need the Python standard library.

## More practice

Stochastic calculus and simulation questions from quant interviews, with worked answers: https://geodesicpartners.co.uk/quant-questions-stochastic-calculus

MIT licence.
