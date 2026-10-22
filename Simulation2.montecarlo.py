# ==================================================================
# Monte Carlo summary for Simulation 2 (Figure 3): replicate datasets
# from the same design, showing that the SSMD-LDA divergence is
# structural (population-level) rather than a property of one seed.
#
# Python translation of Simulation2.montecarlo.R. Python's random-number
# generator differs from R's, so individual replicates differ; the
# medians, percentiles and probabilities agree up to Monte Carlo error.
#
# R -> Python correspondence
#   MASS::mvrnorm        -> numpy RandomState.multivariate_normal
#   glm(binomial)        -> unpenalized logistic regression by IRLS (below)
#   e1071::svm (linear)  -> sklearn.svm.SVC(kernel="linear"); both wrap libsvm
#   pROC::coords(youden) -> Youden threshold over pROC's midpoint thresholds
#   polyroot             -> numpy.roots
#   optimize             -> scipy.optimize.minimize_scalar(method="bounded")
#   quantile (type 7)    -> numpy.quantile (default "linear")
#
# Usage:   python Simulation2.montecarlo.py [replicates]   (default 1000)
# Requires: numpy, scipy, scikit-learn
# ==================================================================
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.svm import SVC

R = int(sys.argv[1]) if len(sys.argv) >= 2 else 1000

mu1 = np.array([1.5, 0.0]); mu2 = np.array([-1.5, 0.0]); m = mu1 - mu2
S1 = np.array([[0.5, -0.8], [-0.8, 3.0]]); S2 = np.array([[4.0, 1.2], [1.2, 1.0]])
n1, n2, pi1, pi2 = 80, 320, 0.2, 0.8


def orient(b, md):
    b = np.asarray(b, float).ravel()
    b = b / np.sqrt(np.sum(b ** 2))
    return -b if np.sum(b * md) < 0 else b

def ang(a, b): return np.degrees(np.arccos(min(1.0, abs(np.sum(a * b)))))
def sdeg(b): return np.degrees(np.arctan2(b[1], b[0]))
def pop_ssmd(b): return np.sum(b * m) / np.sqrt(b @ (S1 + S2) @ b)        # true SSMD

def pop_err(b, c):                                                        # true error of b'x > c
    return (pi1 * norm.cdf((c - np.sum(b * mu1)) / np.sqrt(b @ S1 @ b)) +
            pi2 * norm.sf((c - np.sum(b * mu2)) / np.sqrt(b @ S2 @ b)))

def logistic_direction(X, y, maxit=100, tol=1e-10):
    """Unpenalized logistic regression (IRLS, as glm); intercept dropped."""
    Z = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(Z.shape[1])
    for _ in range(maxit):
        eta = Z @ b
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-12, None)
        b_new = np.linalg.solve(Z.T @ (w[:, None] * Z), Z.T @ (w * eta + (y - p)))
        done = np.max(np.abs(b_new - b)) < tol * (np.max(np.abs(b)) + tol)
        b = b_new
        if done:
            break
    return b[1:]

def svm_direction(X, y, cost=1.0):
    """Linear SVM (C = 1) on standardized features, mapped back to the original scale."""
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    fit = SVC(kernel="linear", C=cost).fit((X - mu) / sd, y)
    return fit.coef_.ravel() / sd

def youden_cutoff(y, s):
    """pROC::coords(roc(y, s, direction = '<'), 'best', best.method = 'youden'):
    first (lowest) of pROC's midpoint thresholds maximizing sens + spec - 1."""
    u = np.unique(s)
    thr = np.concatenate([[-np.inf], (u[1:] + u[:-1]) / 2, [np.inf]])
    cases, ctrls = np.sort(s[y == 1]), np.sort(s[y == 0])
    sens = 1 - np.searchsorted(cases, thr, side="right") / len(cases)   # P(case > t)
    spec = np.searchsorted(ctrls, thr, side="right") / len(ctrls)       # P(control <= t)
    J = sens + spec - 1
    return thr[np.flatnonzero(np.isclose(J, J.max()))[0]]

def cov(X): return np.cov(X, rowvar=False)


## population (true-parameter) directions
rng = np.random.RandomState(1)
Xb = np.vstack([rng.multivariate_normal(mu1, S1, 40000), rng.multivariate_normal(mu2, S2, 160000)])
yb = np.r_[np.ones(40000), np.zeros(160000)]
pop = {"SSMD": orient(np.linalg.solve(S1 + S2, m), m),
       "LDA": orient(np.linalg.solve(pi1 * S1 + pi2 * S2, m), m),
       "Logistic": orient(logistic_direction(Xb, yb), m)}
print("Population (true-parameter) directions:")
for k, b in pop.items():
    print("  %-9s angle %+6.1f deg  SSMD(beta) %.3f  AUROC %.3f"
          % (k, sdeg(b), pop_ssmd(b), norm.cdf(pop_ssmd(b))))
bs = pop["SSMD"]
opt = minimize_scalar(lambda c: pop_err(bs, c), bounds=(np.sum(bs * mu2), np.sum(bs * mu1)),
                      method="bounded")
print("  SSMD direction: midpoint error %.3f, Bayes error %.3f"
      % (pop_err(bs, (np.sum(bs * mu1) + np.sum(bs * mu2)) / 2), opt.fun))


## replicates
def one_replicate(rng):
    X1 = rng.multivariate_normal(mu1, S1, n1); X2 = rng.multivariate_normal(mu2, S2, n2)
    X = np.vstack([X1, X2]); y = np.r_[np.ones(n1, int), np.zeros(n2, int)]
    md = X1.mean(0) - X2.mean(0); s1 = cov(X1); s2 = cov(X2)
    B = [orient(np.linalg.solve(s1 + s2, md), md),
         orient(np.linalg.solve(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2), md), md),
         orient(logistic_direction(X, y), md),
         orient(svm_direction(X, y), md)]
    s = X @ B[0]
    a1, a2 = s[y == 1].mean(), s[y == 0].mean()
    v1, v2 = s[y == 1].var(ddof=1), s[y == 0].var(ddof=1)
    qa = 1 / (2 * v1) - 1 / (2 * v2); qb = -a1 / v1 + a2 / v2
    qd = a1 ** 2 / (2 * v1) - a2 ** 2 / (2 * v2) - np.log((pi1 / np.sqrt(v1)) / (pi2 / np.sqrt(v2)))
    rt = np.real(np.roots([qa, qb, qd]))
    inside = rt[(rt >= a2) & (rt <= a1)]
    bayes = inside[0] if inside.size else np.nan     # R: rt[...][1] is NA when empty
    youden = youden_cutoff(y, s)
    return [ang(B[0], B[1]), ang(B[0], B[2]), ang(B[0], B[3]),
            *[norm.cdf(pop_ssmd(b)) for b in B],
            pop_err(B[0], (a1 + a2) / 2), pop_err(B[0], youden), pop_err(B[0], bayes)]

rng = np.random.RandomState(2026)
D = np.array([one_replicate(rng) for _ in range(R)])

lab = ["angle SSMD-LDA", "angle SSMD-Logistic", "angle SSMD-SVM",
       "true AUROC, SSMD dir", "true AUROC, LDA dir", "true AUROC, Logistic dir", "true AUROC, SVM dir",
       "true error, midpoint", "true error, Youden", "true error, Bayes"]
print("\n%d replicates: median [5th, 95th percentile]" % R)
for j, lb in enumerate(lab):
    q = np.quantile(D[:, j], [0.5, 0.05, 0.95])
    print("  %-26s %.3f [%.3f, %.3f]" % (lb, q[0], q[1], q[2]))
print("  P(SSMD dir has highest true AUROC) = %.3f" % np.mean(D[:, 3] >= D[:, 4:7].max(axis=1)))
print("  P(true AUROC SSMD > LDA) = %.3f;  P(> SVM) = %.3f;  P(> Logistic) = %.3f"
      % (np.mean(D[:, 3] > D[:, 4]), np.mean(D[:, 3] > D[:, 6]), np.mean(D[:, 3] > D[:, 5])))
print("  P(Bayes < Youden < midpoint error) = %.3f"
      % np.mean((D[:, 9] < D[:, 8]) & (D[:, 8] < D[:, 7])))
