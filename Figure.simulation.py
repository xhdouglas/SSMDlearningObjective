#!/usr/bin/env python3
"""
Reproduces Figures 2-4 of
"Strictly Standardized Mean Difference as a Learning Objective for
Interpretable Machine Learning" (SSMD vs Fisher LDA vs logistic regression
vs linear SVM).

    Figure 2: equal covariance, independent groups, balanced   (seed 123)
    Figure 3: heteroscedastic, class-imbalanced                (seed 11)
    Figure 4: paired / correlated groups                       (seed 7)

Notation follows the manuscript: X_g is the n_g x k data matrix of group g,
the projected scores are L_g = X_g beta, and in the paired design the
difference score is d = (X_1 - X_2) beta with SSMD_d = mean(d) / sd(d).

Requires: numpy, scikit-learn, matplotlib.
Usage:    python3 Figure.simulation.py [output_dir]
Output:   figure2.png, figure3.png, figure4.png
"""
import sys
import os
import numpy as np
from numpy.linalg import solve, eigvalsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
from matplotlib.patches import Ellipse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
COLOR_G1 = "#3B7DD8"
COLOR_G2 = "#D9534F"
COLOR_SSMD = "#1f77b4"
COLOR_LDA = "#2ca02c"
COLOR_LOG = "#9467bd"
COLOR_SVM = "#ff7f0e"
COLOR_YOUDEN = "#17becf"
COLOR_BAYES = "#E69500"


# ------------------------------------------------------------- helpers
def orient(b, md):
    """Normalize to unit length and point from G2 toward G1."""
    b = np.asarray(b, dtype=float)
    b = b / np.linalg.norm(b)
    return -b if b @ md < 0 else b


def angle(a, b):
    """Acute angle (degrees) between two unit directions."""
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def logistic_direction(X, y):
    lr = LogisticRegression(C=np.inf, max_iter=10000)  # unpenalized
    lr.fit(X, y)
    return lr.coef_[0]


def svm_direction(X, y, C=1.0):
    """Linear soft-margin SVM on standardized features, mapped back to the
    original feature scale."""
    scaler = StandardScaler().fit(X)
    svm = SVC(kernel="linear", C=C).fit(scaler.transform(X), y)
    return svm.coef_[0] / scaler.scale_


def cov_ellipse(ax, mean, cov, color, nstd=2.0, lw=1.3, alpha=0.8, zorder=2):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=theta, fill=False,
                         color=color, lw=lw, alpha=alpha, zorder=zorder))


def cutline(b, cval, half):
    """Endpoints of the 2-D line {x : b'x = c} (perpendicular to b)."""
    b = b / np.linalg.norm(b)
    p0 = cval * b
    perp = np.array([-b[1], b[0]])
    return p0 - half * perp, p0 + half * perp


def error_rate(s, y, c):
    return float(np.mean((s > c).astype(int) != y))


# ================================================================= FIG 2
def figure2():
    """Independent groups, equal covariance, balanced design."""
    rng = np.random.default_rng(123)
    n = 100
    mu1, mu2 = np.array([1.0, 1.0]), np.array([-1.0, -1.0])
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    X1 = rng.multivariate_normal(mu1, Sigma, n)
    X2 = rng.multivariate_normal(mu2, Sigma, n)
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)

    md = X1.mean(0) - X2.mean(0)
    S1, S2 = np.cov(X1.T, ddof=1), np.cov(X2.T, ddof=1)

    beta_ssmd = orient(solve(S1 + S2, md), md)              # Sigma* = S1+S2
    Sw = ((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2)
    beta_lda = orient(solve(Sw, md), md)                    # pooled covariance
    beta_log = orient(logistic_direction(X, y), md)
    beta_svm = orient(svm_direction(X, y), md)

    methods = [("SSMD", beta_ssmd, COLOR_SSMD), ("LDA", beta_lda, COLOR_LDA),
               ("Logistic", beta_log, COLOR_LOG), ("SVM", beta_svm, COLOR_SVM)]

    res = {}
    for name, b, _ in methods:
        s = X @ b
        c0 = 0.5 * (s[y == 1].mean() + s[y == 0].mean())    # equal-variance midpoint
        pred = (s > c0).astype(int)
        res[name] = dict(
            cut=c0, auc=roc_auc_score(y, s),
            TP=int(((pred == 1) & (y == 1)).sum()),
            TN=int(((pred == 0) & (y == 0)).sum()),
            FP=int(((pred == 1) & (y == 0)).sum()),
            FN=int(((pred == 0) & (y == 1)).sum()))
    ang = {"SSMD-LDA": angle(beta_ssmd, beta_lda),
           "SSMD-Logistic": angle(beta_ssmd, beta_log),
           "SSMD-SVM": angle(beta_ssmd, beta_svm)}
    print("Figure 2 angles:", {k: round(float(v), 1) for k, v in ang.items()})
    for k, v in res.items():
        print("  ", k, {kk: round(float(vv), 3) for kk, vv in v.items()})

    fig = plt.figure(figsize=(12.5, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.scatter(X1[:, 0], X1[:, 1], s=22, alpha=0.55, color=COLOR_G1, label="G$_1$ (class 1)")
    ax.scatter(X2[:, 0], X2[:, 1], s=22, alpha=0.55, color=COLOR_G2, label="G$_2$ (class 0)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    for name, b, c in methods:
        p1, p2 = cutline(b, res[name]["cut"], 6.5)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, lw=1.6, ls="--", alpha=0.85)
        ax.annotate("", xy=tuple(b * 2.6), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
    ax.text(3.05, 3.55, "SSMD", color=COLOR_SSMD, fontsize=12, fontweight="bold")
    ax.text(3.05, 3.15, r"LDA ($=$SSMD)", color=COLOR_LDA, fontsize=12, fontweight="bold")
    ax.text(3.05, 2.75, "Logistic", color=COLOR_LOG, fontsize=12, fontweight="bold")
    ax.text(3.05, 2.35, "SVM", color=COLOR_SVM, fontsize=12, fontweight="bold")
    ax.set_xlim(-5.3, 6.3)
    ax.set_ylim(-4.8, 4.8)
    ax.set_xlabel("X$_1$")
    ax.set_ylabel("X$_2$")
    ax.set_aspect("equal")
    ax.set_title("(a) Projection directions (arrows) and cutoff lines (dashed)\n"
                 f"angles — SSMD–LDA {ang['SSMD-LDA']:.1f}°,  "
                 f"SSMD–Logistic {ang['SSMD-Logistic']:.1f}°,  "
                 f"SSMD–SVM {ang['SSMD-SVM']:.1f}°",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.text(-4.6, -4.3, "dashed = cutoff line (L = c)", fontsize=8, color="grey", style="italic")

    for i, (name, b, _) in enumerate(methods):
        axh = fig.add_subplot(gs[1, i])
        r = res[name]
        s = X @ b
        bins = np.linspace(s.min(), s.max(), 24)
        axh.hist(s[y == 1], bins=bins, alpha=0.6, color=COLOR_G1, label="G$_1$")
        axh.hist(s[y == 0], bins=bins, alpha=0.6, color=COLOR_G2, label="G$_2$")
        axh.axvline(r["cut"], color="black", ls="--", lw=1.3)
        axh.set_title(f"({'bcde'[i]}) {name}   AUROC={r['auc']:.3f}, c={r['cut']:.2f}\n"
                      f"confusion [[TN={r['TN']} FP={r['FP']}] [FN={r['FN']} TP={r['TP']}]]",
                      fontsize=9.5)
        axh.set_xlabel("projected score L", fontsize=9)
        if i == 0:
            axh.set_ylabel("count", fontsize=9)
            axh.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure2.png"), dpi=160, facecolor="white")
    plt.close(fig)


# ================================================================= FIG 3
def figure3():
    """Independent groups, unequal covariance, 1:4 class imbalance."""
    rng = np.random.default_rng(11)
    n1, n2 = 80, 320                                        # imbalanced 1:4
    mu1, mu2 = np.array([1.5, 0.0]), np.array([-1.5, 0.0])
    S1t = np.array([[0.5, 0.0], [0.0, 3.0]])                # tight along x
    S2t = np.array([[4.0, 0.0], [0.0, 0.5]])                # wide along x
    X1 = rng.multivariate_normal(mu1, S1t, n1)
    X2 = rng.multivariate_normal(mu2, S2t, n2)
    X = np.vstack([X1, X2])
    y = np.array([1] * n1 + [0] * n2)

    md = X1.mean(0) - X2.mean(0)
    s1, s2 = np.cov(X1.T, ddof=1), np.cov(X2.T, ddof=1)
    beta_ssmd = orient(solve(s1 + s2, md), md)              # unweighted sum
    Sw = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)    # pooled covariance
    beta_lda = orient(solve(Sw, md), md)
    beta_log = orient(logistic_direction(X, y), md)
    beta_svm = orient(svm_direction(X, y), md)
    ang = {"SSMD-LDA": angle(beta_ssmd, beta_lda),
           "SSMD-Logistic": angle(beta_ssmd, beta_log),
           "SSMD-SVM": angle(beta_ssmd, beta_svm)}
    print("Figure 3 angles:", {k: round(float(v), 1) for k, v in ang.items()})
    for name, b in [("SSMD", beta_ssmd), ("LDA", beta_lda),
                    ("Logistic", beta_log), ("SVM", beta_svm)]:
        sc = X @ b
        ssmd = (sc[y == 1].mean() - sc[y == 0].mean()) / np.sqrt(
            sc[y == 1].var(ddof=1) + sc[y == 0].var(ddof=1))
        print(f"   {name:9s} AUROC={roc_auc_score(y, sc):.3f}  SSMD of score={ssmd:.2f}")

    # --- cutoffs on the SSMD score
    s = X @ beta_ssmd
    mm1, mm2 = s[y == 1].mean(), s[y == 0].mean()
    v1, v2 = s[y == 1].var(ddof=1), s[y == 0].var(ddof=1)
    pi1, pi2 = n1 / (n1 + n2), n2 / (n1 + n2)
    mid = 0.5 * (mm1 + mm2)
    # Bayes (Gaussian score model): a L^2 + b L + d = 0 from log[pi1 f1] = log[pi2 f2]
    a = 1 / (2 * v1) - 1 / (2 * v2)
    b = -mm1 / v1 + mm2 / v2
    d = mm1**2 / (2 * v1) - mm2**2 / (2 * v2) - np.log((pi1 / np.sqrt(v1)) / (pi2 / np.sqrt(v2)))
    roots = np.roots([a, b, d])
    roots = np.real(roots[np.isreal(roots)])
    between = [r for r in roots if min(mm1, mm2) <= r <= max(mm1, mm2)]
    bayes = between[0] if between else roots[0]
    fpr, tpr, thr = roc_curve(y, s)
    youden = thr[np.argmax(tpr - fpr)]                      # Youden's J
    # empirical error minimum over all distinct cutoffs (midpoints of sorted scores)
    ss = np.sort(s)
    cand = np.concatenate([[ss[0] - 1e-9], 0.5 * (ss[:-1] + ss[1:]), [ss[-1] + 1e-9]])
    errs_all = np.array([error_rate(s, y, c) for c in cand])
    emp_min = errs_all.min()
    emp_c = cand[errs_all == emp_min]
    print(f"   score means {mm1:+.2f}, {mm2:+.2f}; score variances {v1:.2f}, {v2:.2f}")

    def conf(c):
        pred = (s > c).astype(int)
        return (int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum()),
                int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum()),
                float(np.mean(pred != y)))
    for nm, c in [("midpoint", mid), ("Youden", youden), ("Bayes", bayes)]:
        tn, fp, fn, tp, err = conf(c)
        print(f"   {nm:9s} c={c:+.2f}  [TN,FP,FN,TP]=[{tn},{fp},{fn},{tp}]  error={err:.3f}")
    print(f"   empirical minimum error={emp_min:.4f} for c in [{emp_c.min():+.2f}, {emp_c.max():+.2f}]")

    methods = [("SSMD", beta_ssmd, COLOR_SSMD), ("LDA", beta_lda, COLOR_LDA),
               ("Logistic", beta_log, COLOR_LOG), ("SVM", beta_svm, COLOR_SVM)]

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.scatter(X2[:, 0], X2[:, 1], s=18, alpha=0.45, color=COLOR_G2, label=f"G$_2$ (class 0, n={n2})")
    ax.scatter(X1[:, 0], X1[:, 1], s=18, alpha=0.55, color=COLOR_G1, label=f"G$_1$ (class 1, n={n1})")
    cov_ellipse(ax, X1.mean(0), s1, COLOR_G1)
    cov_ellipse(ax, X2.mean(0), s2, COLOR_G2)
    for _, bb, c in methods:
        ax.annotate("", xy=tuple(bb * 2.7), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
    ax.text(3.0, 1.55, "SVM", color=COLOR_SVM, fontsize=12, fontweight="bold")
    ax.text(3.0, 1.15, r"SSMD ($\approx$SVM)", color=COLOR_SSMD, fontsize=12, fontweight="bold")
    ax.text(3.0, 0.75, "Logistic", color=COLOR_LOG, fontsize=12, fontweight="bold")
    ax.text(3.0, 0.35, "LDA (pooled)", color=COLOR_LDA, fontsize=12, fontweight="bold")
    p1, p2 = cutline(beta_ssmd, youden, 6.0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COLOR_YOUDEN, lw=2.0, ls="--",
            label=f"Youden cutoff (c={youden:+.2f})")
    p1, p2 = cutline(beta_ssmd, bayes, 6.0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COLOR_BAYES, lw=2.4, ls="-",
            label=f"Bayes cutoff, Gaussian model (c={bayes:+.2f})")
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-5.3, 5.3)
    ax.set_xlabel("X$_1$")
    ax.set_ylabel("X$_2$")
    ax.set_title("(a) Directions (arrows) and the two heteroscedasticity-aware cutoffs on the SSMD axis\n"
                 f"angles — SSMD–LDA {ang['SSMD-LDA']:.1f}°,  "
                 f"SSMD–Logistic {ang['SSMD-Logistic']:.1f}°,  "
                 f"SSMD–SVM {ang['SSMD-SVM']:.1f}°",
                 fontsize=12, fontweight="bold")
    h, l = ax.get_legend_handles_labels()
    order = [1, 0, 2, 3]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=9, framealpha=0.9)

    # (b) score histogram
    axb = fig.add_subplot(gs[1, 0])
    bins = np.linspace(s.min(), s.max(), 26)
    axb.hist(s[y == 1], bins=bins, density=True, alpha=0.6, color=COLOR_G1, label="G$_1$")
    axb.hist(s[y == 0], bins=bins, density=True, alpha=0.6, color=COLOR_G2, label="G$_2$")
    ymax = axb.get_ylim()[1]
    axb.axvline(mid, color="grey", ls=":", lw=1.4)
    axb.text(mid - 0.25, ymax * 0.97, f"midpoint (invalid)\nc={mid:+.2f}", fontsize=7.5,
             color="grey", ha="right", va="top")
    axb.axvline(youden, color=COLOR_YOUDEN, ls="--", lw=1.8)
    axb.text(youden - 0.12, ymax * 0.62, f"Youden\nc={youden:+.2f}", fontsize=8,
             color=COLOR_YOUDEN, ha="right", va="top")
    axb.axvline(bayes, color=COLOR_BAYES, lw=2.2)
    axb.text(bayes + 0.15, ymax * 0.97, f"Bayes\nc={bayes:+.2f}", fontsize=8.5,
             color=COLOR_BAYES, ha="left", va="top", fontweight="bold")
    axb.set_title(f"(b) SSMD score: unequal variances ({v1:.2f} vs {v2:.2f})", fontsize=11)
    axb.set_xlabel("projected score L (SSMD direction)")
    axb.set_ylabel("density")
    axb.legend(fontsize=8, loc="center left")

    # (c) error vs cutoff
    axc = fig.add_subplot(gs[1, 1])
    cs = np.linspace(s.min(), s.max(), 400)
    errs = [error_rate(s, y, c) for c in cs]
    ym = max(errs)
    axc.plot(cs, errs, color="black", lw=1.6, label="overall error (empirical)")
    axc.axvline(mid, color="grey", ls=":", lw=1.2)
    axc.text(mid - 0.3, ym * 0.99, "midpoint", fontsize=8, color="grey", ha="right", va="top")
    axc.axvline(youden, color=COLOR_YOUDEN, ls="--", lw=1.6)
    axc.text(youden - 0.3, ym * 0.87, "Youden", fontsize=8, color=COLOR_YOUDEN, ha="right", va="top")
    axc.axvline(bayes, color=COLOR_BAYES, lw=2.0)
    axc.text(bayes + 0.3, ym * 0.75, "Bayes (Gaussian model)", fontsize=8,
             color=COLOR_BAYES, ha="left", va="top")
    axc.scatter([mid], [error_rate(s, y, mid)], color="grey", zorder=5)
    axc.scatter([youden], [error_rate(s, y, youden)], color=COLOR_YOUDEN, zorder=5,
                label=f"Youden: {error_rate(s, y, youden):.1%}")
    axc.scatter([bayes], [error_rate(s, y, bayes)], color=COLOR_BAYES, zorder=5,
                label=f"Bayes: {error_rate(s, y, bayes):.1%}")
    axc.scatter([emp_c[len(emp_c) // 2]], [emp_min], marker="v", s=55, color="black", zorder=6,
                label=f"empirical minimum: {emp_min:.2%}")
    axc.set_title("(c) Overall error vs cutoff on the SSMD score", fontsize=11)
    axc.set_xlabel("cutoff c")
    axc.set_ylabel("error rate")
    axc.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure3.png"), dpi=160, facecolor="white")
    plt.close(fig)


# ================================================================= FIG 4
def figure4():
    """Paired design: the same n subjects measured under two conditions."""
    rng = np.random.default_rng(7)
    n = 150
    mu1, mu2 = np.array([1.2, 0.3]), np.array([-1.0, -0.2])
    Sigma1 = np.array([[1.0, 0.3], [0.3, 0.6]])
    Sigma2 = np.array([[0.6, -0.2], [-0.2, 1.0]])
    Sigma12 = np.array([[0.5, 0.1], [0.15, 0.4]])           # cross-covariance
    joint_cov = np.block([[Sigma1, Sigma12], [Sigma12.T, Sigma2]])
    assert eigvalsh(joint_cov).min() > 0                    # valid joint covariance
    XY = rng.multivariate_normal(np.concatenate([mu1, mu2]), joint_cov, n)
    X1, X2 = XY[:, :2], XY[:, 2:]                           # row i = same subject

    md = X1.mean(0) - X2.mean(0)
    S1, S2 = np.cov(X1.T, ddof=1), np.cov(X2.T, ddof=1)
    S12 = np.cov(X1.T, X2.T, ddof=1)[:2, 2:]
    # SSMD keeps the cross-covariance: Sigma* = S1 + S2 - S12 - S12'
    beta_ssmd = orient(solve(S1 + S2 - S12 - S12.T, md), md)

    # LDA / logistic / SVM treat the two conditions as independent samples
    X = np.vstack([X1, X2])
    y = np.array([1] * n + [0] * n)
    Sw = ((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2)
    beta_lda = orient(solve(Sw, md), md)
    beta_log = orient(logistic_direction(X, y), md)
    beta_svm = orient(svm_direction(X, y), md)

    def paired_ssmd(b):
        dd = (X1 - X2) @ b
        return dd.mean() / dd.std(ddof=1)
    p_ssmd, p_lda, p_log, p_svm = (paired_ssmd(beta_ssmd), paired_ssmd(beta_lda),
                                   paired_ssmd(beta_log), paired_ssmd(beta_svm))
    a_ssmd_lda, a_ssmd_log, a_ssmd_svm = (angle(beta_ssmd, beta_lda),
                                          angle(beta_ssmd, beta_log),
                                          angle(beta_ssmd, beta_svm))
    print(f"Figure 4 angles: SSMD-LDA {a_ssmd_lda:.1f}, SSMD-Logistic {a_ssmd_log:.1f}, "
          f"SSMD-SVM {a_ssmd_svm:.1f}, LDA-Logistic {angle(beta_lda, beta_log):.1f}")
    print(f"   paired SSMD_d: SSMD {p_ssmd:.2f}, SVM {p_svm:.2f}, "
          f"Logistic {p_log:.2f}, LDA {p_lda:.2f}")

    thetas = np.linspace(0, np.pi, 721)
    vals, vals_indep = [], []
    for t in thetas:
        dvec = np.array([np.cos(t), np.sin(t)])
        dd = (X1 - X2) @ dvec
        vals.append(abs(dd.mean()) / dd.std(ddof=1))
        vals_indep.append(abs(dd.mean()) / np.sqrt(dvec @ (S1 + S2) @ dvec))   # ignores S12
    vals, vals_indep = np.array(vals), np.array(vals_indep)
    deg = lambda b: np.degrees(np.arctan2(b[1], b[0])) % 180
    a_s, a_l, a_g, a_v = deg(beta_ssmd), deg(beta_lda), deg(beta_log), deg(beta_svm)
    i_ind = int(np.argmax(vals_indep))
    print(f"   independence approximation: peak {vals_indep[i_ind]:.2f} at "
          f"{np.degrees(thetas[i_ind]):.1f} deg (LDA {a_l:.1f} deg, SSMD {a_s:.1f} deg)")

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, :])
    for i in range(n):
        ax.plot([X1[i, 0], X2[i, 0]], [X1[i, 1], X2[i, 1]], color="grey", lw=0.35, alpha=0.35, zorder=1)
    ax.scatter(X1[:, 0], X1[:, 1], s=18, alpha=0.6, color=COLOR_G1, label="G$_1$ (condition 1)", zorder=2)
    ax.scatter(X2[:, 0], X2[:, 1], s=18, alpha=0.6, color=COLOR_G2, label="G$_2$ (condition 2)", zorder=2)
    cov_ellipse(ax, X1.mean(0), S1, COLOR_G1)
    cov_ellipse(ax, X2.mean(0), S2, COLOR_G2)
    for bb, c, lw in [(beta_ssmd, COLOR_SSMD, 2.2), (beta_svm, COLOR_SVM, 2.2),
                      (beta_log, COLOR_LOG, 2.6), (beta_lda, COLOR_LDA, 1.6)]:
        ax.annotate("", xy=tuple(bb * 3.2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw), zorder=3)
    ax.text(3.3, 1.55, "SSMD (paired)", color=COLOR_SSMD, fontsize=12, fontweight="bold")
    ax.text(3.3, 1.10, "SVM", color=COLOR_SVM, fontsize=12, fontweight="bold")
    t_lda = ax.text(3.3, 0.65, "LDA ", color=COLOR_LDA, fontsize=12, fontweight="bold")
    t_apx = ax.annotate(r"$\approx$ ", xycoords=t_lda, xy=(1, 0), va="bottom",
                        color="black", fontsize=12, fontweight="bold")
    ax.annotate("Logistic", xycoords=t_apx, xy=(1, 0), va="bottom",
                color=COLOR_LOG, fontsize=12, fontweight="bold")
    ax.text(3.3, 0.20, "(LDA, logistic and SVM ignore pairing)", color="grey", fontsize=10, style="italic")
    ax.set_xlim(-5, 8.3)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("X$_1$")
    ax.set_ylabel("X$_2$")
    ax.set_title("(a) Paired groups (grey lines link the same subject across conditions): "
                 "non-zero correlation, unequal covariance\n"
                 r"SSMD uses the cross-covariance $\Sigma_{12}$ and tilts "
                 f"{a_ssmd_lda:.1f}° from LDA, {a_ssmd_log:.1f}° from logistic and "
                 f"{a_ssmd_svm:.1f}° from SVM",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # (b) paired difference score
    axb = fig.add_subplot(gs[1, 0])
    D_s, D_l, D_v = (X1 - X2) @ beta_ssmd, (X1 - X2) @ beta_lda, (X1 - X2) @ beta_svm
    bins = np.linspace(min(D_s.min(), D_l.min(), D_v.min()),
                       max(D_s.max(), D_l.max(), D_v.max()), 22)
    axb.hist(D_s, bins=bins, density=True, alpha=0.55, color=COLOR_SSMD,
             label=f"SSMD direction: SSMD$_d$={p_ssmd:.2f}")
    axb.hist(D_v, bins=bins, density=True, alpha=0.45, color=COLOR_SVM,
             label=f"SVM direction: SSMD$_d$={p_svm:.2f}")
    axb.hist(D_l, bins=bins, density=True, alpha=0.50, color=COLOR_LDA,
             label=f"LDA direction: SSMD$_d$={p_lda:.2f}")
    axb.axvline(D_s.mean(), color=COLOR_SSMD, lw=1.8)
    axb.axvline(D_v.mean(), color=COLOR_SVM, lw=1.8, ls=":")
    axb.axvline(D_l.mean(), color=COLOR_LDA, lw=1.8, ls="--")
    axb.axvline(0, color="black", ls=":", lw=1.0, alpha=0.6)
    axb.set_title(r"(b) Paired difference score $\mathbf{d} = (\mathbf{X}_1-\mathbf{X}_2)\,\beta$",
                  fontsize=11)
    axb.set_xlabel("paired difference score d")
    axb.set_ylabel("density")
    axb.legend(fontsize=8, loc="upper left")

    # (c) paired SSMD vs direction angle
    axc = fig.add_subplot(gs[1, 1])
    axc.plot(np.degrees(thetas), vals, color="black", lw=1.8,
             label=r"paired SSMD$_d$ (uses $\Sigma_{12}$)")
    axc.plot(np.degrees(thetas), vals_indep, color="grey", lw=1.5, ls="--",
             label=r"independence approximation (ignores $\Sigma_{12}$)")
    for ang_, c in [(a_s, COLOR_SSMD), (a_v, COLOR_SVM), (a_g, COLOR_LOG), (a_l, COLOR_LDA)]:
        axc.axvline(ang_, color=c, lw=1.4, alpha=0.85)
    axc.scatter([a_s], [p_ssmd], color=COLOR_SSMD, zorder=5, s=40, label=f"SSMD ({a_s:.1f}°)")
    axc.scatter([a_v], [p_svm], color=COLOR_SVM, zorder=5, s=40, label=f"SVM ({a_v:.1f}°)")
    axc.scatter([a_l], [p_lda], color=COLOR_LDA, zorder=5, s=40, label=f"LDA ({a_l:.1f}°)")
    axc.scatter([a_g], [p_log], color=COLOR_LOG, zorder=5, s=40, label=f"Logistic ({a_g:.1f}°)")
    axc.scatter([np.degrees(thetas[i_ind])], [vals_indep[i_ind]], marker="v", color="grey",
                zorder=5, s=45, label=f"independence peak ({vals_indep[i_ind]:.2f})")
    axc.set_xlim(0, 180)
    axc.set_title(r"(c) Only the SSMD direction maximizes the paired SSMD$_d$", fontsize=11)
    axc.set_xlabel("projection-direction angle (degrees)")
    axc.set_ylabel(r"paired SSMD$_d$")
    axc.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure4.png"), dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    print("saved figure2.png, figure3.png, figure4.png to", os.path.abspath(OUT))
