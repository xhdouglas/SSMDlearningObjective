# ==================================================================
# Python companion code for Figures 2-4 of
# "Strictly Standardized Mean Difference as a Learning Objective for
#  Interpretable Machine Learning"
#  (SSMD vs Fisher LDA vs logistic regression vs linear SVM)
#
#   Figure 2: equal covariance, independent groups, balanced
#   Figure 3: heteroscedastic, oppositely tilted covariances, imbalanced (1:4)
#   Figure 4: paired / correlated groups
#
# Line-by-line Python translation of Figure.simulation.R: same designs,
# same estimators, same cutoffs, same panels and the same export
# specification. Python's random-number generator differs from R's, so the
# simulated points -- and therefore the exact numbers (angles, AUROC,
# cutoffs, errors) -- differ from the R figures; the qualitative pattern is
# the same.
#
# Notation follows the manuscript: X_g is the n_g x k data matrix of
# group g, the projected scores are L_g = X_g beta, and in the paired
# design the difference score is d = (X_1 - X_2) beta with
# SSMD_d = mean(d) / sd(d).
#
# R -> Python correspondence
#   MASS::mvrnorm        -> numpy RandomState.multivariate_normal
#   glm(binomial)        -> unpenalized logistic regression by IRLS (below)
#   e1071::svm (linear)  -> sklearn.svm.SVC(kernel="linear"); both wrap libsvm
#   pROC::auc            -> Mann-Whitney AUROC (below)
#   pROC::coords(youden) -> Youden threshold over pROC's midpoint thresholds
#   polyroot             -> numpy.roots
#   ggplot2 / gridExtra  -> matplotlib (GridSpec, constrained layout)
#   ragg / cairo_pdf     -> matplotlib Agg + Pillow (TIFF, LZW) / PDF backend
#
# Usage:   python Figure.simulation.py [output_dir]
# Output:  figure2_py, figure3_py, figure4_py as .tiff (183 mm wide, 600 dpi,
#          LZW), .pdf (vector) and .png (preview); see save_fig() below
# Needs:   numpy, scipy, scikit-learn, matplotlib, pillow
#
# Seeds: 123 (Fig. 2), 35 (Fig. 3), 7 (Fig. 4) with numpy's legacy
# RandomState generator (the seeds named for the Python figures in the
# header of the R script).
# ==================================================================
import os
import sys

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) >= 2 else "."
os.makedirs(OUT, exist_ok=True)

## ---------------------------------------------------------------- style
COL_G1, COL_G2 = "#3B7DD8", "#D9534F"
COL_SSMD, COL_LDA, COL_LOG, COL_SVM = "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"
COL_YOUDEN, COL_BAYES = "#17becf", "#E69500"
GREY = {k: "#%02x%02x%02x" % ((round(255 * k / 100),) * 3) for k in (20, 30, 35, 40, 45, 60, 80)}
METHOD_COLS = {"SSMD": COL_SSMD, "LDA": COL_LDA, "Logistic": COL_LOG, "SVM": COL_SVM}
GROUP_COLS = {1: COL_G1, 0: COL_G2}
METHODS = ["SSMD", "LDA", "Logistic", "SVM"]

## ggplot2 unit conversions: text size is in mm, linewidth 1 = 0.75 mm
PT = 72.27 / 25.4
def gtext(size): return size * PT                    # geom_text size -> points
def glw(lw): return lw * 0.75 * PT                   # linewidth -> points
def gpt(size): return ((size * PT + 0.5 * 96 / 25.4 / 2) * 0.75) ** 2   # point size -> scatter s

## theme_bw(base_size = 10) without grid lines, centred bold titles
plt.rcParams.update({
    "font.family": "DejaVu Sans",          # has the degree, approx-equal and Sigma glyphs
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": GREY[20], "axes.linewidth": 0.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.color": GREY[20], "ytick.color": GREY[20],
    "xtick.labelcolor": GREY[30], "ytick.labelcolor": GREY[30],
    "legend.fontsize": 8, "legend.frameon": False, "legend.fancybox": False,
    "legend.edgecolor": GREY[80], "legend.framealpha": 1,
    "axes.grid": False, "mathtext.default": "regular",
    "pdf.fonttype": 42,
})

## -------------------------------------------------------------- helpers
def orient(b, md):
    """Normalize to unit length and point from G2 toward G1 (fixes sign ambiguity)."""
    b = np.asarray(b, float).ravel()
    b = b / np.sqrt(np.sum(b ** 2))
    return -b if np.sum(b * md) < 0 else b

def angle_deg(a, b):
    """Acute angle (degrees) between two unit directions."""
    return np.degrees(np.arccos(min(1.0, abs(np.sum(a * b)))))

def dir_deg(b):
    """Direction angle in [0, 180) degrees."""
    return np.degrees(np.arctan2(b[1], b[0])) % 180

def logistic_direction(X, y, maxit=100, tol=1e-10):
    """Unpenalized logistic regression (IRLS, as glm) direction, intercept dropped."""
    Z = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(Z.shape[1])
    for _ in range(maxit):
        eta = Z @ b
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-12, None)
        z = eta + (y - p) / w
        b_new = np.linalg.solve(Z.T @ (w[:, None] * Z), Z.T @ (w * z))
        if np.max(np.abs(b_new - b)) < tol * (np.max(np.abs(b)) + tol):
            b = b_new
            break
        b = b_new
    return b[1:]

def svm_direction(X, y, cost=1.0):
    """Linear soft-margin SVM (C = 1) on standardized features, mapped back to
    the original feature scale (as e1071::svm on scale(X))."""
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    fit = SVC(kernel="linear", C=cost).fit((X - mu) / sd, y)
    return fit.coef_.ravel() / sd

def ellipse_xy(center, S, nstd=2, npts=200):
    """2-SD covariance ellipse."""
    vals, vecs = np.linalg.eigh(S)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    tt = np.linspace(0, 2 * np.pi, npts)
    xy = (vecs @ np.diag(np.sqrt(vals)) @ np.vstack([np.cos(tt), np.sin(tt)])).T * nstd
    return center[0] + xy[:, 0], center[1] + xy[:, 1]

def cutline(b, cval, half=6):
    """Cutoff in score space, L = c, as the 2-D line {x : b'x = c};
    it is perpendicular to b and passes through c*b (b unit-norm)."""
    p0 = cval * b
    perp = np.array([-b[1], b[0]])
    return [p0[0] - half * perp[0], p0[0] + half * perp[0]], \
           [p0[1] - half * perp[1], p0[1] + half * perp[1]]

def draw_arrow(ax, b, length, color, lw=1.0):
    """Arrow from the origin along b (closed head, 0.18 cm, 30 degrees)."""
    ax.annotate("", xy=(b[0] * length, b[1] * length), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.23",
                                mutation_scale=0.18 / 2.54 * 72 / 0.4, color=color,
                                lw=glw(lw), shrinkA=0, shrinkB=0))

def expand(lo, hi, mult=0.05):
    """ggplot2's default 5% expansion of continuous axis limits."""
    r = hi - lo
    return lo - mult * r, hi + mult * r

def gg_bins(values, bins):
    """Bin edges as in ggplot2's geom_histogram(bins = k)."""
    lo, hi = np.min(values), np.max(values)
    w = (hi - lo) / (bins - 1)
    return lo - w / 2 + w * np.arange(bins + 1)

def tag(ax, letter):
    """ggplot2 plot tag at the top-left corner of the panel."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 20),
                textcoords="offset points", ha="left", va="bottom", fontsize=12)

def boxed(leg):
    leg.get_frame().set_edgecolor(GREY[80])
    leg.get_frame().set_linewidth(0.6)
    return leg

## ---------------------------------------------------------------- export
## Output follows the Nature Portfolio figure specifications:
##   * width  = 183 mm (double column); height <= 170 mm (room for legend)
##   * raster resolution >= 300 dpi at final size -- we use 600 dpi, the
##     usual standard for graphs that contain text and thin lines
##   * text 5-7 pt at final size
## Each figure is laid out on a canvas of FINAL_WIDTH_MM / TEXT_SCALE
## (about 256 mm) and printed at 183 mm, so all text and lines shrink by
## TEXT_SCALE: the 7-pt legend text becomes 5 pt and the 10-pt panel titles
## about 7 pt. The TIFF is written at the final size and resolution
## (about 4323 pixels wide, resolution tag 600 dpi, LZW compression, RGB):
## the canvas is rasterized at DPI * TEXT_SCALE and tagged as 600 dpi.
## A vector PDF -- the format Nature prefers for graphs -- and a PNG preview
## are written alongside.
FINAL_WIDTH_MM = 183
MAX_HEIGHT_MM = 170
DPI = 600
TEXT_SCALE = 0.714

def new_figure(width, height):
    """Canvas of FINAL_WIDTH_MM / TEXT_SCALE with the design aspect ratio."""
    h_mm = FINAL_WIDTH_MM * height / width
    w_in = FINAL_WIDTH_MM / TEXT_SCALE / 25.4
    h_in = h_mm / TEXT_SCALE / 25.4
    fig = plt.figure(figsize=(w_in, h_in), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.04)
    return fig, h_mm

def save_fig(stem, fig, h_mm):
    if h_mm > MAX_HEIGHT_MM:
        raise ValueError("%s: final height %.0f mm exceeds %d mm"
                         % (os.path.basename(stem), h_mm, MAX_HEIGHT_MM))
    ## vector PDF: page drawn at FINAL_WIDTH_MM / TEXT_SCALE, to be placed at 183 mm
    fig.savefig(stem + ".pdf", facecolor="white")
    ## raster at final size: canvas pixels = final mm at 600 dpi
    fig.savefig(stem + ".png", dpi=DPI * TEXT_SCALE, facecolor="white")
    plt.close(fig)
    img = Image.open(stem + ".png").convert("RGB")
    ## TIFF at final size, 600 dpi, LZW compression; PNG preview with identical pixels
    img.save(stem + ".tiff", compression="tiff_lzw", dpi=(DPI, DPI))
    img.save(stem + ".png", dpi=(DPI, DPI))
    print("   %s: %.0f x %.0f mm at %d dpi = %d x %d pixels"
          % (os.path.basename(stem), FINAL_WIDTH_MM, h_mm, DPI, img.size[0], img.size[1]))

def error_rate(s, y, cval): return np.mean((s > cval).astype(int) != y)

def auroc(y, s):
    """AUROC with cases (y = 1) expected higher (pROC direction '<')."""
    r = rankdata(s)
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    return (np.sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)

def youden_cutoff(y, s):
    """pROC::coords(roc, 'best', best.method = 'youden'): thresholds are
    midpoints between consecutive distinct scores (plus +-Inf); returns the
    first (lowest) threshold maximizing sensitivity + specificity - 1."""
    u = np.unique(s)
    thr = np.concatenate([[-np.inf], (u[1:] + u[:-1]) / 2, [np.inf]])
    cases, ctrls = s[y == 1], s[y == 0]
    sens = np.array([np.mean(cases > t) for t in thr])
    spec = np.array([np.mean(ctrls <= t) for t in thr])
    J = sens + spec - 1
    return thr[np.flatnonzero(np.isclose(J, J.max()))[0]]

def confusion(s, y, cval):
    pred = (s > cval).astype(int)
    return {"TN": int(np.sum((pred == 0) & (y == 0))), "FP": int(np.sum((pred == 1) & (y == 0))),
            "FN": int(np.sum((pred == 0) & (y == 1))), "TP": int(np.sum((pred == 1) & (y == 1)))}

def cm_str(cm): return ",".join(str(cm[k]) for k in ("TN", "FP", "FN", "TP"))

def cov(X): return np.cov(X, rowvar=False)

# ==================================================================
# FIGURE 2 -- independent groups, equal covariance, balanced design
# ==================================================================
def figure2(seed=123):
    rng = np.random.RandomState(seed)
    n = 100
    mu1, mu2 = np.array([1.0, 1.0]), np.array([-1.0, -1.0])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    X1 = rng.multivariate_normal(mu1, Sigma, n)
    X2 = rng.multivariate_normal(mu2, Sigma, n)
    X = np.vstack([X1, X2])
    y = np.r_[np.ones(n, int), np.zeros(n, int)]           # 1 = G1, 0 = G2

    md = X1.mean(0) - X2.mean(0)
    S1, S2 = cov(X1), cov(X2)
    ## SSMD: independent groups, so Sigma* = S1 + S2 (cross-covariance = 0)
    beta = {
        "SSMD": orient(np.linalg.solve(S1 + S2, md), md),
        "LDA": orient(np.linalg.solve(((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2), md), md),
        "Logistic": orient(logistic_direction(X, y), md),
        "SVM": orient(svm_direction(X, y), md)}

    res = {}
    for m in METHODS:
        s = X @ beta[m]
        c0 = 0.5 * (s[y == 1].mean() + s[y == 0].mean())  # equal-variance midpoint
        res[m] = dict(method=m, score=s, cut=c0, auc=auroc(y, s), cm=confusion(s, y, c0))
    ang = {m: angle_deg(beta["SSMD"], beta[m]) for m in ("LDA", "Logistic", "SVM")}
    print("Figure 2 angles: SSMD-LDA %.1f, SSMD-Logistic %.1f, SSMD-SVM %.1f"
          % (ang["LDA"], ang["Logistic"], ang["SVM"]))
    for r in res.values():
        print("   %-9s c=%+.3f AUROC=%.3f  [TN,FP,FN,TP]=[%s]"
              % (r["method"], r["cut"], r["auc"], cm_str(r["cm"])))

    fig, h_mm = new_figure(12.5, 9.6)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.15, 1])

    ## ---- panel (a)
    ax = fig.add_subplot(gs[0, :])
    ax.axhline(0, color=GREY[60], lw=glw(0.3), zorder=0)
    ax.axvline(0, color=GREY[60], lw=glw(0.3), zorder=0)
    for g, lab in ((1, "$G_1$ (class 1)"), (0, "$G_2$ (class 0)")):
        ax.scatter(X[y == g, 0], X[y == g, 1], s=gpt(1.4), color=GROUP_COLS[g],
                   alpha=0.55, linewidths=0, label=lab)
    for m in METHODS:
        xs, ys = cutline(beta[m], res[m]["cut"], 6.5)
        ax.plot(xs, ys, ls="--", color=METHOD_COLS[m], lw=glw(0.6))
    for m in METHODS:
        draw_arrow(ax, beta[m], 2.6, METHOD_COLS[m], 1.0)
    for yy, lab, m in zip([3.55, 3.15, 2.75, 2.35],
                          ["SSMD", "LDA (= SSMD)", "Logistic", "SVM"], METHODS):
        ax.text(3.05, yy, lab, color=METHOD_COLS[m], ha="left", va="center",
                fontweight="bold", fontsize=gtext(3.6))
    ax.set_xlim(*expand(-5.3, 6.3)); ax.set_ylim(*expand(-4.8, 4.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$X_1$"); ax.set_ylabel("$X_2$")
    ax.set_title("(a) Projection directions (arrows) and cutoff lines (dashed)\n"
                 "angles - SSMD-LDA %.1f°,  SSMD-Logistic %.1f°,  SSMD-SVM %.1f°"
                 % (ang["LDA"], ang["Logistic"], ang["SVM"]))
    boxed(ax.legend(loc="lower right", frameon=True, markerscale=1.5))

    ## ---- panels (b)-(e): projected-score histograms
    def hist_panel(ax, r, tg):
        edges = gg_bins(r["score"], 23)
        for g in (0, 1):                                 # factor levels "0", "1"
            ax.hist(r["score"][y == g], bins=edges, color=GROUP_COLS[g], alpha=0.6,
                    label="$G_1$" if g == 1 else "$G_2$")
        ax.axvline(r["cut"], ls="--", color="black", lw=glw(0.5))
        ax.set_xlabel("projected score L")
        if tg == "b":
            ax.set_ylabel("count")
            h, l = ax.get_legend_handles_labels()
            ax.legend(h[::-1], l[::-1], loc="upper right")
        cm = r["cm"]
        ax.set_title("(%s) %s  AUROC=%.3f, c=%.2f\n[TN=%d FP=%d] [FN=%d TP=%d]"
                     % (tg, r["method"], r["auc"], r["cut"], cm["TN"], cm["FP"], cm["FN"], cm["TP"]),
                     fontsize=8.5, fontweight="normal")
    for j, (m, tg) in enumerate(zip(METHODS, "bcde")):
        hist_panel(fig.add_subplot(gs[1, j]), res[m], tg)

    save_fig(os.path.join(OUT, "figure2_py"), fig, h_mm)

# ==================================================================
# FIGURE 3 -- independent groups, unequal and oppositely tilted
# covariances, 1:4 imbalance. The population directions of SSMD,
# (Sigma1 + Sigma2)^-1 m, and pooled LDA, Sw^-1 m, differ by about 24 deg.
# ==================================================================
FIG3_SEED = 35
def figure3(seed=FIG3_SEED):
    rng = np.random.RandomState(seed)
    n1, n2 = 80, 320
    mu1, mu2 = np.array([1.5, 0.0]), np.array([-1.5, 0.0])
    S1t = np.array([[0.5, -0.8], [-0.8, 3.0]])        # tight along X1, tilted
    S2t = np.array([[4.0, 1.2], [1.2, 1.0]])          # wide along X1, opposite tilt
    X1 = rng.multivariate_normal(mu1, S1t, n1)
    X2 = rng.multivariate_normal(mu2, S2t, n2)
    X = np.vstack([X1, X2])
    y = np.r_[np.ones(n1, int), np.zeros(n2, int)]

    md = X1.mean(0) - X2.mean(0)
    s1, s2 = cov(X1), cov(X2)
    ## SSMD uses the unweighted SUM; LDA uses the sample-size-weighted POOLED covariance
    beta = {
        "SSMD": orient(np.linalg.solve(s1 + s2, md), md),
        "LDA": orient(np.linalg.solve(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2), md), md),
        "Logistic": orient(logistic_direction(X, y), md),
        "SVM": orient(svm_direction(X, y), md)}
    ang = {m: angle_deg(beta["SSMD"], beta[m]) for m in ("LDA", "Logistic", "SVM")}
    print("Figure 3 angles: SSMD-LDA %.1f, SSMD-Logistic %.1f, SSMD-SVM %.1f"
          % (ang["LDA"], ang["Logistic"], ang["SVM"]))

    def ssmd_of(b):
        sc = X @ b
        return (sc[y == 1].mean() - sc[y == 0].mean()) / \
            np.sqrt(sc[y == 1].var(ddof=1) + sc[y == 0].var(ddof=1))
    def sdeg(b): return np.degrees(np.arctan2(b[1], b[0]))   # signed direction angle
    st = {m: dict(auc=auroc(y, X @ beta[m]), ssmd=ssmd_of(beta[m]), deg=sdeg(beta[m]))
          for m in METHODS}
    for m in METHODS:
        print("   %-9s AUROC=%.3f  SSMD of score=%.2f  angle=%+.1f deg"
              % (m, st[m]["auc"], st[m]["ssmd"], st[m]["deg"]))
    ## population values: AUROC = Phi(SSMD(beta)) for independent normal groups
    for m in METHODS:
        b = beta[m]
        sp = np.sum(b * (mu1 - mu2)) / np.sqrt(b @ (S1t + S2t) @ b)
        print("   %-9s population SSMD(beta)=%.3f  population AUROC=%.3f" % (m, sp, norm.cdf(sp)))

    ## ---- cutoffs on the SSMD score
    s = X @ beta["SSMD"]
    mm1, mm2 = s[y == 1].mean(), s[y == 0].mean()
    v1, v2 = s[y == 1].var(ddof=1), s[y == 0].var(ddof=1)
    pi1, pi2 = n1 / (n1 + n2), n2 / (n1 + n2)
    mid = 0.5 * (mm1 + mm2)                           # equal-variance midpoint
    ## Bayes rule under a Gaussian score model: classify as G1 when
    ## pi1 f1(L) > pi2 f2(L); taking logs gives a L^2 + b L + d = 0 and the
    ## cutoff is the root between the two score means
    a = 1 / (2 * v1) - 1 / (2 * v2)
    b = -mm1 / v1 + mm2 / v2
    d0 = mm1 ** 2 / (2 * v1) - mm2 ** 2 / (2 * v2) - np.log((pi1 / np.sqrt(v1)) / (pi2 / np.sqrt(v2)))
    roots = np.real(np.roots([a, b, d0]))
    bayes = roots[(roots >= min(mm1, mm2)) & (roots <= max(mm1, mm2))][0]
    ## (cost-sensitive variant: replace pi2/pi1 by the FP/FN cost ratio)
    ## Youden cutoff: distribution-free, maximizes sensitivity + specificity - 1
    youden = youden_cutoff(y, s)
    ## empirical minimum of the overall error over all distinct cutoffs
    so = np.sort(s)
    cand = np.concatenate([[so[0] - 1e-9], (so[1:] + so[:-1]) / 2, [so[-1] + 1e-9]])
    errs_all = np.array([error_rate(s, y, cc) for cc in cand])
    emp_min = errs_all.min()
    emp_c = cand[errs_all == emp_min]
    emp_pt = emp_c[int(np.ceil(len(emp_c) / 2)) - 1]
    print("   score means %+.2f, %+.2f; score variances %.2f, %.2f" % (mm1, mm2, v1, v2))
    for nm, cc in (("midpoint", mid), ("Youden", youden), ("Bayes", bayes)):
        print("   %-9s c=%+.2f  [TN,FP,FN,TP]=[%s]  error=%.3f"
              % (nm, cc, cm_str(confusion(s, y, cc)), error_rate(s, y, cc)))
    print("   empirical minimum error=%.4f for c in [%+.2f, %+.2f]"
          % (emp_min, emp_c.min(), emp_c.max()))

    fig, h_mm = new_figure(15, 11.4)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.3], width_ratios=[1, 1, 1.12])

    ## ---- panel a
    ax = fig.add_subplot(gs[0, :])
    hg2 = ax.scatter(X2[:, 0], X2[:, 1], s=gpt(1.2), color=COL_G2, alpha=0.45, linewidths=0)
    hg1 = ax.scatter(X1[:, 0], X1[:, 1], s=gpt(1.2), color=COL_G1, alpha=0.55, linewidths=0)
    ax.plot(*ellipse_xy(X1.mean(0), s1), color=COL_G1, lw=glw(0.5))
    ax.plot(*ellipse_xy(X2.mean(0), s2), color=COL_G2, lw=glw(0.5))
    for m in METHODS:
        draw_arrow(ax, beta[m], 3.2, METHOD_COLS[m], 1.0)
    hy, = ax.plot(*cutline(beta["SSMD"], youden, 6.0), ls="--", color=COL_YOUDEN, lw=glw(0.9))
    hb, = ax.plot(*cutline(beta["SSMD"], bayes, 6.0), ls="-", color=COL_BAYES, lw=glw(0.9))
    ord_ = sorted(METHODS, key=lambda m: -st[m]["deg"])   # label order = arrow order, top to bottom
    lab_txt = {"SSMD": "SSMD  (S1+S2)^-1 m", "LDA": "LDA  Sp^-1 m (pooled)",
               "Logistic": "Logistic", "SVM": "SVM"}
    for i, m in enumerate(ord_):
        ax.text(3.5, 2.3 - 0.75 * i, "%s  (%+.1f°)" % (lab_txt[m], st[m]["deg"]),
                color=METHOD_COLS[m], ha="left", va="center", fontweight="bold",
                fontsize=gtext(3.4))
    ax.set_xlim(*expand(-8, 8.5)); ax.set_ylim(*expand(-5.3, 6.2))
    ax.set_xlabel("$X_1$"); ax.set_ylabel("$X_2$")
    ax.set_title("angles - SSMD-LDA %.1f°,  SSMD-Logistic %.1f°,  SSMD-SVM %.1f°"
                 % (ang["LDA"], ang["Logistic"], ang["SVM"]), fontsize=11, fontweight="normal")
    tag(ax, "a")
    g1h = Line2D([], [], ls="", marker="o", color=COL_G1, ms=5)
    g2h = Line2D([], [], ls="", marker="o", color=COL_G2, ms=5)
    boxed(ax.legend([g1h, g2h, hy, hb],
                    ["G1 (class 1, n=%d)" % n1, "G2 (class 0, n=%d)" % n2,
                     "Youden cutoff (c=%+.2f)" % youden,
                     "Bayes cutoff, Gaussian model (c=%+.2f)" % bayes],
                    ncol=2, loc="upper left", bbox_to_anchor=(0.008, 0.985),
                    frameon=True, fontsize=9, columnspacing=1.2, handlelength=2.2))

    ## ---- panel b: score histograms with the three cutoffs
    ax = fig.add_subplot(gs[1, 0])
    edges = gg_bins(s, 25)
    dmax = 0
    for g in (0, 1):
        h, _, _ = ax.hist(s[y == g], bins=edges, density=True, color=GROUP_COLS[g], alpha=0.6,
                          label="$G_1$" if g == 1 else "$G_2$")
        dmax = max(dmax, h.max())
    ax.axvline(mid, color=GREY[45], ls=":", lw=glw(0.7))
    ax.axvline(youden, color=COL_YOUDEN, ls="--", lw=glw(0.8))
    ax.axvline(bayes, color=COL_BAYES, lw=glw(1.0))
    yr = (0 - 0.02 * dmax, dmax + 0.30 * dmax)
    ax.set_ylim(*yr)
    ax.set_xlim(*expand(edges[0], edges[-1]))
    top = blended_transform_factory(ax.transData, ax.transAxes)
    ## y = Inf with vjust > 1: the text top sits (vjust - 1) text-heights below the top
    for x0, vj, fs, col, lab in ((mid - 0.2, 1.3, 2.6, GREY[45], "midpoint\nc=%+.2f" % mid),
                                 (youden - 0.1, 4.5, 2.8, COL_YOUDEN, "Youden\nc=%+.2f" % youden)):
        hgt = 2 * gtext(fs) * 1.1
        ax.annotate(lab, xy=(x0, 1), xycoords=top, xytext=(0, -(vj - 1) * hgt),
                    textcoords="offset points", ha="right", va="top", color=col,
                    fontsize=gtext(fs), linespacing=0.95)
    ## Bayes label: just right of the Bayes line, in the headroom above the bars
    ax.text(bayes + 0.12, yr[1] - 0.03 * (yr[1] - yr[0]), "Bayes\nc=%+.2f" % bayes,
            ha="left", va="top", color=COL_BAYES, fontsize=gtext(2.9), fontweight="bold",
            linespacing=0.95)
    h, l = ax.get_legend_handles_labels()
    boxed(ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0.015, 0.985),
                    frameon=True, fontsize=9))
    ax.set_xlabel("projected score L (SSMD direction)"); ax.set_ylabel("density")
    ax.set_title("SSMD score: variances %.2f vs %.2f" % (v1, v2), fontweight="normal")
    tag(ax, "b")

    ## ---- panel c: overall error vs cutoff
    ax = fig.add_subplot(gs[1, 1])
    cs = np.linspace(s.min(), s.max(), 400)
    ax.plot(cs, [error_rate(s, y, cc) for cc in cs], color="black", lw=glw(0.7))
    ax.axvline(mid, color=GREY[45], ls=":", lw=glw(0.5))
    ax.axvline(youden, color=COL_YOUDEN, ls="--", lw=glw(0.5))
    ax.axvline(bayes, color=COL_BAYES, lw=glw(0.9))
    pts = [(mid, error_rate(s, y, mid), GREY[45], "o", "midpoint: %.1f%%" % (100 * error_rate(s, y, mid))),
           (youden, error_rate(s, y, youden), COL_YOUDEN, "o", "Youden: %.1f%%" % (100 * error_rate(s, y, youden))),
           (bayes, error_rate(s, y, bayes), COL_BAYES, "o", "Bayes: %.1f%%" % (100 * error_rate(s, y, bayes))),
           (emp_pt, emp_min, "black", "v", "minimum: %.2f%%" % (100 * emp_min))]
    for xx, ee, col, mk, lab in pts:
        ax.scatter([xx], [ee], s=gpt(2.4), color=col, marker=mk, linewidths=0, zorder=5, label=lab)
    ax.set_xlabel("cutoff c"); ax.set_ylabel("error rate")
    ax.set_title("Overall error vs cutoff (SSMD score)", fontweight="normal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9,
              markerscale=1.3, columnspacing=0.8, handletextpad=0.3)
    tag(ax, "c")

    ## ---- panel d: SSMD and AUROC of the projected score vs direction angle
    ax = fig.add_subplot(gs[1, 2])
    th = np.arange(-60, 30 + 1e-9, 0.25)
    vs = np.column_stack([np.cos(np.radians(th)), np.sin(np.radians(th))])
    scan_ssmd = np.array([ssmd_of(v) for v in vs])
    scan_auc = np.array([auroc(y, X @ v) for v in vs])
    ## map AUROC onto the SSMD axis for a secondary axis
    r_s = (scan_ssmd.min(), scan_ssmd.max()); r_a = (scan_auc.min(), scan_auc.max())
    k = (r_s[1] - r_s[0]) / (r_a[1] - r_a[0])
    to_s = lambda a_: r_s[0] + (np.asarray(a_) - r_a[0]) * k
    from_s = lambda s_: r_a[0] + (np.asarray(s_) - r_s[0]) / k
    l1, = ax.plot(th, scan_ssmd, color="black", lw=glw(0.8), label="SSMD (left axis)")
    l2, = ax.plot(th, to_s(scan_auc), color=GREY[45], ls="--", lw=glw(0.8), label="AUROC (right axis)")
    mh = []
    for m in METHODS:
        ax.axvline(st[m]["deg"], color=METHOD_COLS[m], lw=glw(0.6))
    for m in METHODS:
        mh.append(ax.scatter([st[m]["deg"]], [st[m]["ssmd"]], s=gpt(2.6), color=METHOD_COLS[m],
                             edgecolors="white", linewidths=0.8, zorder=5,
                             label="%s %.2f / %.3f" % (m, st[m]["ssmd"], st[m]["auc"])))
    ax.set_xlim(-60, 30)
    ax.set_ylim(*expand(min(scan_ssmd.min(), to_s(scan_auc).min()),
                        max(scan_ssmd.max(), to_s(scan_auc).max())))
    sec = ax.secondary_yaxis("right", functions=(from_s, to_s))
    sec.set_ylabel("AUROC")
    ax.set_xlabel("projection-direction angle (degrees)"); ax.set_ylabel("SSMD")
    ax.set_title("SSMD and AUROC vs direction", fontweight="normal")
    leg1 = ax.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.5, -0.17),
                     ncol=2, fontsize=9, handlelength=3, columnspacing=1.0)
    ax.add_artist(leg1)
    ax.legend(handles=mh, loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=2, fontsize=9,
              title="direction: SSMD / AUROC", title_fontsize=9, markerscale=1.3,
              columnspacing=0.8, handletextpad=0.3)
    tag(ax, "d")

    save_fig(os.path.join(OUT, "figure3_py"), fig, h_mm)

# ==================================================================
# FIGURE 4 -- paired design: the same n subjects under two conditions
# ==================================================================
def figure4(seed=7):
    rng = np.random.RandomState(seed)
    n = 150
    mu1, mu2 = np.array([1.2, 0.3]), np.array([-1.0, -0.2])
    Sigma1 = np.array([[1.0, 0.3], [0.3, 0.6]])
    Sigma2 = np.array([[0.6, -0.2], [-0.2, 1.0]])
    Sigma12 = np.array([[0.5, 0.1], [0.15, 0.4]])
    joint = np.block([[Sigma1, Sigma12], [Sigma12.T, Sigma2]])
    assert np.min(np.linalg.eigvalsh(joint)) > 0
    XY = rng.multivariate_normal(np.r_[mu1, mu2], joint, n)
    X1, X2 = XY[:, :2], XY[:, 2:]                     # row i = same subject

    md = X1.mean(0) - X2.mean(0)
    S1, S2 = cov(X1), cov(X2)
    S12 = np.cov(X1, X2, rowvar=False)[:2, 2:]
    ## SSMD keeps the cross-covariance: Sigma* = S1 + S2 - S12 - S12'
    b_ssmd = orient(np.linalg.solve(S1 + S2 - S12 - S12.T, md), md)
    ## LDA / logistic / SVM treat the two conditions as independent samples
    X = np.vstack([X1, X2])
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    beta = {
        "SSMD": b_ssmd,
        "LDA": orient(np.linalg.solve(((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2), md), md),
        "Logistic": orient(logistic_direction(X, y), md),
        "SVM": orient(svm_direction(X, y), md)}

    def paired_ssmd(b):
        dd = (X1 - X2) @ b
        return dd.mean() / dd.std(ddof=1)
    pss = {m: paired_ssmd(beta[m]) for m in METHODS}
    ang = {m: angle_deg(beta["SSMD"], beta[m]) for m in ("LDA", "Logistic", "SVM")}
    print("Figure 4 angles: SSMD-LDA %.1f, SSMD-Logistic %.1f, SSMD-SVM %.1f, LDA-Logistic %.1f"
          % (ang["LDA"], ang["Logistic"], ang["SVM"], angle_deg(beta["LDA"], beta["Logistic"])))
    print("   paired SSMD_d: SSMD %.2f, SVM %.2f, Logistic %.2f, LDA %.2f"
          % (pss["SSMD"], pss["SVM"], pss["Logistic"], pss["LDA"]))

    ## paired SSMD_d (true, uses S12) and the independence approximation
    ## (ignores S12) as functions of the direction angle
    th = np.linspace(0, np.pi, 721)
    paired, indep = [], []
    for t in th:
        v = np.array([np.cos(t), np.sin(t)])
        dd = (X1 - X2) @ v
        paired.append(abs(dd.mean()) / dd.std(ddof=1))
        indep.append(abs(dd.mean()) / np.sqrt(v @ (S1 + S2) @ v))
    deg = np.degrees(th); paired = np.array(paired); indep = np.array(indep)
    i_ind = int(np.argmax(indep))
    dd_ang = {m: dir_deg(beta[m]) for m in METHODS}
    print("   independence approximation: peak %.2f at %.1f deg (LDA %.1f deg, SSMD %.1f deg)"
          % (indep[i_ind], deg[i_ind], dd_ang["LDA"], dd_ang["SSMD"]))

    fig, h_mm = new_figure(13, 10.2)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.15])

    ## ---- panel (a)
    ax = fig.add_subplot(gs[0, :])
    segs = np.stack([X1, X2], axis=1)
    ax.add_collection(LineCollection(segs, colors=GREY[60], linewidths=glw(0.2), alpha=0.5))
    ax.scatter(X1[:, 0], X1[:, 1], s=gpt(1.2), color=COL_G1, alpha=0.6, linewidths=0,
               label="G1 (condition 1)")
    ax.scatter(X2[:, 0], X2[:, 1], s=gpt(1.2), color=COL_G2, alpha=0.6, linewidths=0,
               label="G2 (condition 2)")
    ax.plot(*ellipse_xy(X1.mean(0), S1), color=COL_G1, lw=glw(0.5))
    ax.plot(*ellipse_xy(X2.mean(0), S2), color=COL_G2, lw=glw(0.5))
    for m in ("SSMD", "SVM", "Logistic", "LDA"):
        draw_arrow(ax, beta[m], 3.2, METHOD_COLS[m], 0.7 if m == "LDA" else 1.0)
    for xx, yy, lab, col, style in (
            (3.3, 1.55, "SSMD (paired)", COL_SSMD, "normal"), (3.3, 1.10, "SVM", COL_SVM, "normal"),
            (3.3, 0.65, "LDA", COL_LDA, "normal"), (3.95, 0.65, "≈", "black", "normal"),
            (4.25, 0.65, "Logistic", COL_LOG, "normal")):
        ax.text(xx, yy, lab, color=col, ha="left", va="center", fontweight="bold",
                fontsize=gtext(3.8))
    ax.text(3.3, 0.20, "(LDA, logistic and SVM ignore pairing)", color=GREY[45], ha="left",
            va="center", fontstyle="italic", fontsize=gtext(3.3))
    ax.set_xlim(*expand(-5, 8.3)); ax.set_ylim(*expand(-5, 5))
    ax.set_xlabel("$X_1$"); ax.set_ylabel("$X_2$")
    ax.set_title("(a) Paired groups (grey lines link the same subject across conditions): "
                 "non-zero correlation, unequal covariance\n"
                 "SSMD uses the cross-covariance Σ₁₂ and tilts %.1f° from LDA, %.1f° from "
                 "logistic and %.1f° from SVM" % (ang["LDA"], ang["Logistic"], ang["SVM"]))
    boxed(ax.legend(loc="lower right", frameon=True, markerscale=1.5))

    ## ---- panel (b) paired difference score d = (X1 - X2) beta
    ax = fig.add_subplot(gs[1, 0])
    bm = ["SSMD", "SVM", "LDA"]
    dvals = {m: (X1 - X2) @ beta[m] for m in bm}
    edges = gg_bins(np.concatenate(list(dvals.values())), 21)
    dmax = 0
    for m, col in zip(bm, (COL_SSMD, COL_SVM, COL_LDA)):
        h, _, _ = ax.hist(dvals[m], bins=edges, density=True, color=col, alpha=0.5,
                          label="%s: SSMD_d = %.2f" % (m, pss[m]))
        dmax = max(dmax, h.max())
    for m, col, ls in zip(bm, (COL_SSMD, COL_SVM, COL_LDA), ("-", ":", "--")):
        ax.axvline(dvals[m].mean(), color=col, ls=ls, lw=glw(0.8))
    ax.axvline(0, ls=":", color=GREY[40], lw=glw(0.5))
    ax.set_ylim(-0.02 * dmax, dmax * 1.10)
    ax.set_xlim(*expand(edges[0], edges[-1]))
    boxed(ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.97), frameon=True, fontsize=9))
    ax.set_xlabel("paired difference score d"); ax.set_ylabel("density")
    ax.set_title(r"(b) Paired difference score $\mathbf{d} = (\mathbf{X}_1 - \mathbf{X}_2)\,\beta$",
                 fontweight="normal")

    ## ---- panel (c) paired SSMD_d vs direction angle
    ax = fig.add_subplot(gs[1, 1])
    l1, = ax.plot(deg, paired, color="black", lw=glw(0.8), label="paired SSMD_d (uses Σ₁₂)")
    l2, = ax.plot(deg, indep, color=GREY[45], ls="--", lw=glw(0.8),
                  label="independence approximation (ignores Σ₁₂)")
    cm4 = ["SSMD", "SVM", "LDA", "Logistic"]
    cols4 = [COL_SSMD, COL_SVM, COL_LDA, COL_LOG]
    for m, col in zip(cm4, cols4):
        ax.axvline(dd_ang[m], color=col, lw=glw(0.6), alpha=0.85)
    mh = [ax.scatter([dd_ang[m]], [pss[m]], s=gpt(2.6), color=col, edgecolors="white",
                     linewidths=0.8, zorder=5, label="%s (%.1f°)" % (m, dd_ang[m]))
          for m, col in zip(cm4, cols4)]
    ax.scatter([deg[i_ind]], [indep[i_ind]], marker="v", s=gpt(2.6), color=GREY[45], zorder=5)
    ax.text(deg[i_ind] + 4, indep[i_ind] - 0.15, "independence peak (%.2f)" % indep[i_ind],
            ha="left", va="center", fontsize=gtext(2.7), color=GREY[35])
    ax.set_xlim(0, 180)
    lo = min(paired.min(), indep.min()); hi = max(paired.max(), indep.max())
    ax.set_ylim(*expand(lo, hi))
    ax.set_xlabel("projection-direction angle (degrees)"); ax.set_ylabel("paired SSMD$_d$")
    ax.set_title("(c) Only the SSMD direction maximizes the paired SSMD$_d$", fontweight="normal")
    leg1 = ax.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.5, -0.17),
                     ncol=1, fontsize=9.5, handlelength=3.2)
    ax.add_artist(leg1)
    ax.legend(handles=mh, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2,
              fontsize=9.5, markerscale=1.3, columnspacing=0.8, handletextpad=0.3)

    save_fig(os.path.join(OUT, "figure4_py"), fig, h_mm)


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    print("saved figure2_py, figure3_py, figure4_py (.tiff, .pdf, .png) to", os.path.abspath(OUT))
