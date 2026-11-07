import random
import unittest

from corr import cholesky, correlated_normals, sample_cov


def matmul_t(L):
    n = len(L)
    return [[sum(L[i][k] * L[j][k] for k in range(n)) for j in range(n)] for i in range(n)]


class TestCholesky(unittest.TestCase):
    def test_reconstructs_the_matrix(self):
        a = [[4.0, 2.0, 0.6], [2.0, 5.0, 1.5], [0.6, 1.5, 3.0]]
        L = cholesky(a)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertEqual(L[i][j], 0.0)
        back = matmul_t(L)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(back[i][j], a[i][j], places=12)

    def test_two_by_two_closed_form(self):
        rho = 0.6
        L = cholesky([[1.0, rho], [rho, 1.0]])
        self.assertAlmostEqual(L[1][0], rho, places=15)
        self.assertAlmostEqual(L[1][1], (1 - rho * rho) ** 0.5, places=15)

    def test_rejects_impossible_correlations(self):
        # pairwise 0.9, 0.9 and -0.9 cannot all hold at once: the matrix has a negative eigenvalue
        with self.assertRaises(ValueError):
            cholesky([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
        with self.assertRaises(ValueError):
            cholesky([[1, 0.5], [0.4, 1]])


class TestSampling(unittest.TestCase):
    def test_sample_covariance_matches(self):
        cov = [[1.0, 0.7, -0.3], [0.7, 2.0, 0.2], [-0.3, 0.2, 0.5]]
        xs = correlated_normals(cov, 100_000, mean=[1.0, -2.0, 0.0], rng=random.Random(8))
        got = sample_cov(xs)
        for i in range(3):
            self.assertAlmostEqual(sum(x[i] for x in xs) / len(xs), [1.0, -2.0, 0.0][i], delta=0.02)
            for j in range(3):
                self.assertAlmostEqual(got[i][j], cov[i][j], delta=0.03)


if __name__ == "__main__":
    unittest.main()
