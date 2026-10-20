# ==================================================================
# R companion code for Figures 2-4 of
# "Strictly Standardized Mean Difference as a Learning Objective for
#  Interpretable Machine Learning"
#  (SSMD vs Fisher LDA vs logistic regression vs linear SVM)
#
#   Figure 2: equal covariance, independent groups, balanced
#   Figure 3: heteroscedastic, class-imbalanced (1:4)
#   Figure 4: paired / correlated groups
#
# The published figures were drawn with the Python script
# (Figure.simulation.py). This R script implements the same designs and
# methods with R's random-number generator, so the simulated points --
# and therefore the exact numbers (angles, AUROC, cutoffs, errors) --
# differ slightly from the published figures; the qualitative pattern is
# the same.
#
# Notation follows the manuscript: X_g is the n_g x k data matrix of
# group g, the projected scores are L_g = X_g beta, and in the paired
# design the difference score is d = (X_1 - X_2) beta with
# SSMD_d = mean(d) / sd(d).
#
# Usage:   Rscript Figure.simulation.R [output_dir]
#          (run in a UTF-8 locale -- the default in R on macOS/Windows --
#           so that the degree, approx-equal and Sigma glyphs render)
# Output:  figure2_R.png, figure3_R.png, figure4_R.png
# ==================================================================
suppressPackageStartupMessages({
  library(MASS)       # mvrnorm
  library(ggplot2)
  library(gridExtra)  # arrangeGrob
  library(e1071)      # linear SVM
  library(pROC)       # AUROC and Youden cutoff
})

args <- commandArgs(trailingOnly = TRUE)
OUT  <- if (length(args) >= 1) args[1] else "."
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

## ---------------------------------------------------------------- style
COL_G1 <- "#3B7DD8"; COL_G2 <- "#D9534F"
COL_SSMD <- "#1f77b4"; COL_LDA <- "#2ca02c"; COL_LOG <- "#9467bd"; COL_SVM <- "#ff7f0e"
COL_YOUDEN <- "#17becf"; COL_BAYES <- "#E69500"
METHOD_COLS <- c(SSMD = COL_SSMD, LDA = COL_LDA, Logistic = COL_LOG, SVM = COL_SVM)
GROUP_COLS  <- c("1" = COL_G1, "0" = COL_G2)
## Font with the degree, approx-equal and Sigma/subscript glyphs; falls back to "sans"
FONT <- "sans"
if (requireNamespace("systemfonts", quietly = TRUE) &&
    "DejaVu Sans" %in% systemfonts::system_fonts()$family) FONT <- "DejaVu Sans"
update_geom_defaults("text", list(family = FONT))
## arrangeGrob() measures text on the current device; open an off-screen
## bitmap device so that measurement uses the same font engine as the output
if (requireNamespace("ragg", quietly = TRUE)) {
  ragg::agg_png(tempfile(fileext = ".png"))
} else {
  png(tempfile(fileext = ".png"))
}
theme_set(theme_bw(base_size = 10, base_family = FONT) +
          theme(panel.grid = element_blank(),
                plot.title = element_text(size = 10, face = "bold", hjust = 0.5)))

## -------------------------------------------------------------- helpers
## Normalize to unit length and point from G2 toward G1 (fixes sign ambiguity)
orient <- function(b, md) {
  b <- as.numeric(b) / sqrt(sum(b^2))
  if (sum(b * md) < 0) b <- -b
  b
}
## Acute angle (degrees) between two unit directions
angle_deg <- function(a, b) acos(min(1, abs(sum(a * b)))) * 180 / pi
## Direction angle in [0, 180) degrees
dir_deg <- function(b) (atan2(b[2], b[1]) * 180 / pi) %% 180

## Unpenalized logistic regression direction (intercept dropped)
logistic_direction <- function(X, y) {
  fit <- suppressWarnings(glm(y ~ X, family = binomial))
  as.numeric(coef(fit)[-1])
}
## Linear soft-margin SVM (C = 1) on standardized features, mapped back to
## the original feature scale
svm_direction <- function(X, y, cost = 1) {
  Xsc <- scale(X)
  fit <- svm(Xsc, factor(y), kernel = "linear", cost = cost, scale = FALSE)
  w   <- as.numeric(t(fit$coefs) %*% fit$SV)
  w / attr(Xsc, "scaled:scale")
}
## 2-SD covariance ellipse as a data frame
ellipse_df <- function(center, S, nstd = 2, npts = 200) {
  e  <- eigen(S, symmetric = TRUE)
  tt <- seq(0, 2 * pi, length.out = npts)
  xy <- t(e$vectors %*% diag(sqrt(e$values)) %*% rbind(cos(tt), sin(tt))) * nstd
  data.frame(x = center[1] + xy[, 1], y = center[2] + xy[, 2])
}
## Cutoff in score space, L = c, as the 2-D line {x : b'x = c};
## it is perpendicular to b and passes through c*b (b unit-norm)
cutline_df <- function(b, cval, half = 6) {
  p0 <- cval * b; perp <- c(-b[2], b[1])
  data.frame(x = p0[1] - half * perp[1], y = p0[2] - half * perp[2],
             xend = p0[1] + half * perp[1], yend = p0[2] + half * perp[2])
}
## Arrow from the origin along b
arrow_df <- function(b, len, method)
  data.frame(x = 0, y = 0, xend = b[1] * len, yend = b[2] * len, method = method)
## PNG device: 'ragg' (if installed) gives reliable Unicode glyphs (degree,
## approx-equal, Sigma with subscripts); otherwise fall back to the default
save_png <- function(file, g, width, height) {
  dev <- if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png else "png"
  ggsave(file, g, width = width, height = height, dpi = 160, bg = "white", device = dev)
}
error_rate <- function(s, y, cval) mean(as.integer(s > cval) != y)
auroc <- function(y, s) as.numeric(pROC::auc(y, s, quiet = TRUE, direction = "<"))
confusion <- function(s, y, cval) {
  pred <- as.integer(s > cval)
  c(TN = sum(pred == 0 & y == 0), FP = sum(pred == 1 & y == 0),
    FN = sum(pred == 0 & y == 1), TP = sum(pred == 1 & y == 1))
}

# ==================================================================
# FIGURE 2 -- independent groups, equal covariance, balanced design
# ==================================================================
figure2 <- function() {
  set.seed(123)
  n     <- 100
  mu1   <- c(1, 1); mu2 <- c(-1, -1)
  Sigma <- matrix(c(1, 0.5, 0.5, 1), 2)
  X1 <- mvrnorm(n, mu1, Sigma); X2 <- mvrnorm(n, mu2, Sigma)
  X  <- rbind(X1, X2); y <- c(rep(1, n), rep(0, n))   # 1 = G1, 0 = G2

  md <- colMeans(X1) - colMeans(X2)
  S1 <- cov(X1); S2 <- cov(X2)
  ## SSMD: independent groups, so Sigma* = S1 + S2 (cross-covariance = 0)
  beta <- list(
    SSMD     = orient(solve(S1 + S2, md), md),
    LDA      = orient(solve(((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2), md), md),
    Logistic = orient(logistic_direction(X, y), md),
    SVM      = orient(svm_direction(X, y), md))

  res <- lapply(names(beta), function(m) {
    s  <- as.numeric(X %*% beta[[m]])
    c0 <- 0.5 * (mean(s[y == 1]) + mean(s[y == 0]))  # equal-variance midpoint
    list(method = m, score = s, cut = c0, auc = auroc(y, s), cm = confusion(s, y, c0))
  })
  names(res) <- names(beta)
  ang <- sapply(c("LDA", "Logistic", "SVM"), function(m) angle_deg(beta$SSMD, beta[[m]]))
  cat(sprintf("Figure 2 angles: SSMD-LDA %.1f, SSMD-Logistic %.1f, SSMD-SVM %.1f\n",
              ang["LDA"], ang["Logistic"], ang["SVM"]))
  for (r in res) cat(sprintf("   %-9s c=%+.3f AUROC=%.3f  [TN,FP,FN,TP]=[%s]\n",
                             r$method, r$cut, r$auc, paste(r$cm, collapse = ",")))

  df <- data.frame(x1 = X[, 1], x2 = X[, 2], g = factor(y))
  arrows <- do.call(rbind, lapply(names(beta), function(m) arrow_df(beta[[m]], 2.6, m)))
  cuts   <- do.call(rbind, lapply(names(beta), function(m)
    cbind(cutline_df(beta[[m]], res[[m]]$cut, 6.5), method = m)))
  labs <- data.frame(x = 3.05, y = c(3.55, 3.15, 2.75, 2.35),
                     lab = c("SSMD", "LDA (= SSMD)", "Logistic", "SVM"),
                     method = c("SSMD", "LDA", "Logistic", "SVM"))
  pa <- ggplot(df, aes(x1, x2)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.3) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.3) +
    geom_point(aes(colour = g), alpha = 0.55, size = 1.4) +
    geom_segment(data = cuts, aes(x, y, xend = xend, yend = yend, linetype = "cut"),
                 colour = METHOD_COLS[cuts$method], linewidth = 0.6, show.legend = FALSE) +
    geom_segment(data = arrows, aes(x, y, xend = xend, yend = yend),
                 colour = METHOD_COLS[arrows$method], linewidth = 1.0,
                 arrow = arrow(length = unit(0.18, "cm"), type = "closed")) +
    geom_text(data = labs, aes(x, y, label = lab), colour = METHOD_COLS[labs$method],
              hjust = 0, fontface = "bold", size = 3.6) +
    scale_colour_manual(values = GROUP_COLS, breaks = c("1", "0"),
                        labels = c(expression(G[1]~"(class 1)"), expression(G[2]~"(class 0)")),
                        name = NULL) +
    scale_linetype_manual(values = c(cut = "dashed")) +
    coord_fixed(xlim = c(-5.3, 6.3), ylim = c(-4.8, 4.8)) +
    labs(x = expression(X[1]), y = expression(X[2]),
         title = sprintf("(a) Projection directions (arrows) and cutoff lines (dashed)\nangles - SSMD-LDA %.1f°,  SSMD-Logistic %.1f°,  SSMD-SVM %.1f°",
                         ang["LDA"], ang["Logistic"], ang["SVM"])) +
    theme(legend.position = c(0.88, 0.1), legend.background = element_rect(colour = "grey80"))

  hist_panel <- function(r, tag) {
    d <- data.frame(score = r$score, g = factor(y))
    ggplot(d, aes(score, fill = g)) +
      geom_histogram(alpha = 0.6, position = "identity", bins = 23) +
      geom_vline(xintercept = r$cut, linetype = "dashed") +
      scale_fill_manual(values = GROUP_COLS, breaks = c("1", "0"),
                        labels = c(expression(G[1]), expression(G[2])), name = NULL) +
      labs(x = "projected score L", y = if (tag == "b") "count" else NULL,
           title = sprintf("(%s) %s  AUROC=%.3f, c=%.2f\n[TN=%d FP=%d] [FN=%d TP=%d]",
                           tag, r$method, r$auc, r$cut, r$cm["TN"], r$cm["FP"],
                           r$cm["FN"], r$cm["TP"])) +
      theme(legend.position = if (tag == "b") c(0.85, 0.85) else "none",
            plot.title = element_text(size = 8.5, face = "plain"))
  }
  g <- arrangeGrob(pa,
                   hist_panel(res$SSMD, "b"), hist_panel(res$LDA, "c"),
                   hist_panel(res$Logistic, "d"), hist_panel(res$SVM, "e"),
                   layout_matrix = rbind(c(1, 1, 1, 1), c(2, 3, 4, 5)), heights = c(1.15, 1))
  save_png(file.path(OUT, "figure2_R.png"), g, width = 12.5, height = 9.6)
}

# ==================================================================
# FIGURE 3 -- independent groups, unequal covariance, 1:4 imbalance
# ==================================================================
figure3 <- function() {
  set.seed(11)
  n1 <- 80; n2 <- 320
  mu1 <- c(1.5, 0); mu2 <- c(-1.5, 0)
  S1t <- matrix(c(0.5, 0, 0, 3.0), 2)          # tight along X1
  S2t <- matrix(c(4.0, 0, 0, 0.5), 2)          # wide along X1
  X1 <- mvrnorm(n1, mu1, S1t); X2 <- mvrnorm(n2, mu2, S2t)
  X  <- rbind(X1, X2); y <- c(rep(1, n1), rep(0, n2))

  md <- colMeans(X1) - colMeans(X2)
  s1 <- cov(X1); s2 <- cov(X2)
  ## SSMD uses the unweighted SUM; LDA uses the sample-size-weighted POOLED covariance
  beta <- list(
    SSMD     = orient(solve(s1 + s2, md), md),
    LDA      = orient(solve(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2), md), md),
    Logistic = orient(logistic_direction(X, y), md),
    SVM      = orient(svm_direction(X, y), md))
  ang <- sapply(c("LDA", "Logistic", "SVM"), function(m) angle_deg(beta$SSMD, beta[[m]]))
  cat(sprintf("Figure 3 angles: SSMD-LDA %.1f, SSMD-Logistic %.1f, SSMD-SVM %.1f\n",
              ang["LDA"], ang["Logistic"], ang["SVM"]))
  for (m in names(beta)) {
    sc <- as.numeric(X %*% beta[[m]])
    ss <- (mean(sc[y == 1]) - mean(sc[y == 0])) / sqrt(var(sc[y == 1]) + var(sc[y == 0]))
    cat(sprintf("   %-9s AUROC=%.3f  SSMD of score=%.2f\n", m, auroc(y, sc), ss))
  }

  ## ---- cutoffs on the SSMD score
  s   <- as.numeric(X %*% beta$SSMD)
  mm1 <- mean(s[y == 1]); mm2 <- mean(s[y == 0])
  v1  <- var(s[y == 1]);  v2  <- var(s[y == 0])
  pi1 <- n1 / (n1 + n2);  pi2 <- n2 / (n1 + n2)
  mid <- 0.5 * (mm1 + mm2)                     # equal-variance midpoint
  ## Bayes rule under a Gaussian score model: classify as G1 when
  ## pi1 f1(L) > pi2 f2(L); taking logs gives a L^2 + b L + d = 0 and the
  ## cutoff is the root between the two score means
  a  <- 1 / (2 * v1) - 1 / (2 * v2)
  b  <- -mm1 / v1 + mm2 / v2
  d0 <- mm1^2 / (2 * v1) - mm2^2 / (2 * v2) - log((pi1 / sqrt(v1)) / (pi2 / sqrt(v2)))
  roots <- Re(polyroot(c(d0, b, a)))
  bayes <- roots[roots >= min(mm1, mm2) & roots <= max(mm1, mm2)][1]
  ## (cost-sensitive variant: replace pi2/pi1 by the FP/FN cost ratio)
  ## Youden cutoff: distribution-free, maximizes sensitivity + specificity - 1
  roc_s  <- pROC::roc(y, s, quiet = TRUE, direction = "<")
  youden <- as.numeric(unlist(pROC::coords(roc_s, "best", best.method = "youden",
                                    ret = "threshold")))[1]
  ## empirical minimum of the overall error over all distinct cutoffs
  so   <- sort(s)
  cand <- c(so[1] - 1e-9, (so[-1] + so[-length(so)]) / 2, so[length(so)] + 1e-9)
  errs_all <- sapply(cand, function(cc) error_rate(s, y, cc))
  emp_min  <- min(errs_all); emp_c <- cand[errs_all == emp_min]
  emp_pt   <- emp_c[ceiling(length(emp_c) / 2)]
  cat(sprintf("   score means %+.2f, %+.2f; score variances %.2f, %.2f\n", mm1, mm2, v1, v2))
  for (nm in c("midpoint", "Youden", "Bayes")) {
    cc <- switch(nm, midpoint = mid, Youden = youden, Bayes = bayes)
    cat(sprintf("   %-9s c=%+.2f  [TN,FP,FN,TP]=[%s]  error=%.3f\n", nm, cc,
                paste(confusion(s, y, cc), collapse = ","), error_rate(s, y, cc)))
  }
  cat(sprintf("   empirical minimum error=%.4f for c in [%+.2f, %+.2f]\n",
              emp_min, min(emp_c), max(emp_c)))

  ## ---- panel (a)
  df <- data.frame(x1 = X[, 1], x2 = X[, 2], g = factor(y))
  arrows <- do.call(rbind, lapply(names(beta), function(m) arrow_df(beta[[m]], 2.7, m)))
  cut_y  <- cbind(cutline_df(beta$SSMD, youden, 6.0),
                  type = sprintf("Youden cutoff (c=%+.2f)", youden))
  cut_b  <- cbind(cutline_df(beta$SSMD, bayes, 6.0),
                  type = sprintf("Bayes cutoff, Gaussian model (c=%+.2f)", bayes))
  cuts <- rbind(cut_y, cut_b)
  cuts$type <- factor(cuts$type, levels = c(cut_y$type, cut_b$type))
  labs <- data.frame(x = 3.0, y = c(1.55, 1.15, 0.75, 0.35),
                     lab = c("SVM", "SSMD (≈ SVM)", "Logistic", "LDA (pooled)"),
                     method = c("SVM", "SSMD", "Logistic", "LDA"))
  pa <- ggplot() +
    geom_point(data = df[df$g == "0", ], aes(x1, x2, colour = g), alpha = 0.45, size = 1.2) +
    geom_point(data = df[df$g == "1", ], aes(x1, x2, colour = g), alpha = 0.55, size = 1.2) +
    geom_path(data = ellipse_df(colMeans(X1), s1), aes(x, y), colour = COL_G1) +
    geom_path(data = ellipse_df(colMeans(X2), s2), aes(x, y), colour = COL_G2) +
    geom_segment(data = arrows, aes(x, y, xend = xend, yend = yend),
                 colour = METHOD_COLS[arrows$method], linewidth = 1.0,
                 arrow = arrow(length = unit(0.18, "cm"), type = "closed")) +
    geom_segment(data = cuts, aes(x, y, xend = xend, yend = yend, linetype = type),
                 colour = ifelse(grepl("Youden", cuts$type), COL_YOUDEN, COL_BAYES),
                 linewidth = 0.9) +
    geom_text(data = labs, aes(x, y, label = lab), colour = METHOD_COLS[labs$method],
              hjust = 0, fontface = "bold", size = 3.6) +
    scale_colour_manual(values = GROUP_COLS, breaks = c("1", "0"),
                        labels = c(sprintf("G1 (class 1, n=%d)", n1),
                                   sprintf("G2 (class 0, n=%d)", n2)), name = NULL) +
    scale_linetype_manual(values = c("dashed", "solid"), name = NULL) +
    coord_cartesian(xlim = c(-7.5, 7.5), ylim = c(-5.3, 5.3)) +
    labs(x = expression(X[1]), y = expression(X[2]),
         title = sprintf("(a) Directions (arrows) and the two heteroscedasticity-aware cutoffs on the SSMD axis\nangles - SSMD-LDA %.1f°,  SSMD-Logistic %.1f°,  SSMD-SVM %.1f°",
                         ang["LDA"], ang["Logistic"], ang["SVM"])) +
    theme(legend.position = c(0.17, 0.82), legend.background = element_rect(colour = "grey80"),
          legend.spacing.y = unit(0, "pt"))

  ## ---- panel (b) score histograms with the three cutoffs
  ds <- data.frame(score = s, g = factor(y))
  pb <- ggplot(ds, aes(score, fill = g)) +
    geom_histogram(aes(y = after_stat(density)), alpha = 0.6, position = "identity", bins = 25) +
    geom_vline(xintercept = mid, colour = "grey45", linetype = "dotted", linewidth = 0.7) +
    geom_vline(xintercept = youden, colour = COL_YOUDEN, linetype = "dashed", linewidth = 0.8) +
    geom_vline(xintercept = bayes, colour = COL_BAYES, linewidth = 1.0) +
    annotate("text", x = mid - 0.2, y = Inf, vjust = 1.3, hjust = 1, size = 2.6,
             colour = "grey45", label = sprintf("midpoint (invalid)\nc=%+.2f", mid)) +
    annotate("text", x = youden - 0.1, y = Inf, vjust = 4.5, hjust = 1, size = 2.8,
             colour = COL_YOUDEN, label = sprintf("Youden\nc=%+.2f", youden)) +
    annotate("text", x = bayes + 0.15, y = Inf, vjust = 1.3, hjust = 0, size = 2.9,
             colour = COL_BAYES, fontface = "bold", label = sprintf("Bayes\nc=%+.2f", bayes)) +
    scale_fill_manual(values = GROUP_COLS, breaks = c("1", "0"),
                      labels = c(expression(G[1]), expression(G[2])), name = NULL) +
    labs(x = "projected score L (SSMD direction)", y = "density",
         title = sprintf("(b) SSMD score: unequal variances (%.2f vs %.2f)", v1, v2)) +
    theme(legend.position = c(0.1, 0.5), plot.title = element_text(face = "plain", size = 10))

  ## ---- panel (c) overall error vs cutoff
  cs <- seq(min(s), max(s), length.out = 400)
  de <- data.frame(c = cs, err = sapply(cs, function(cc) error_rate(s, y, cc)))
  pts <- data.frame(c = c(youden, bayes, emp_pt),
                    err = c(error_rate(s, y, youden), error_rate(s, y, bayes), emp_min),
                    what = factor(c(sprintf("Youden: %.1f%%", 100 * error_rate(s, y, youden)),
                                    sprintf("Bayes: %.1f%%", 100 * error_rate(s, y, bayes)),
                                    sprintf("empirical minimum: %.2f%%", 100 * emp_min)),
                                  levels = c(sprintf("Youden: %.1f%%", 100 * error_rate(s, y, youden)),
                                             sprintf("Bayes: %.1f%%", 100 * error_rate(s, y, bayes)),
                                             sprintf("empirical minimum: %.2f%%", 100 * emp_min))))
  ym <- max(de$err)
  pc <- ggplot(de, aes(c, err)) +
    geom_line(linewidth = 0.7) +
    geom_vline(xintercept = mid, colour = "grey45", linetype = "dotted") +
    geom_vline(xintercept = youden, colour = COL_YOUDEN, linetype = "dashed") +
    geom_vline(xintercept = bayes, colour = COL_BAYES, linewidth = 0.9) +
    annotate("text", x = mid - 0.3, y = ym, hjust = 1, vjust = 1, size = 2.8,
             colour = "grey45", label = "midpoint") +
    annotate("text", x = youden - 0.3, y = ym * 0.87, hjust = 1, vjust = 1, size = 2.8,
             colour = COL_YOUDEN, label = "Youden") +
    annotate("text", x = bayes + 0.3, y = ym * 0.75, hjust = 0, vjust = 1, size = 2.8,
             colour = COL_BAYES, label = "Bayes (Gaussian model)") +
    geom_point(data = data.frame(c = mid, err = error_rate(s, y, mid)), colour = "grey45", size = 2) +
    geom_point(data = pts, aes(colour = what, shape = what), size = 2.4) +
    scale_colour_manual(values = c(COL_YOUDEN, COL_BAYES, "black"), name = NULL) +
    scale_shape_manual(values = c(16, 16, 25), name = NULL) +
    labs(x = "cutoff c", y = "error rate", title = "(c) Overall error vs cutoff on the SSMD score") +
    theme(legend.position = c(0.8, 0.85), legend.background = element_rect(colour = "grey80"),
          plot.title = element_text(face = "plain", size = 10))

  g <- arrangeGrob(pa, pb, pc, layout_matrix = rbind(c(1, 1), c(2, 3)), heights = c(1.3, 1))
  save_png(file.path(OUT, "figure3_R.png"), g, width = 13, height = 9.2)
}

# ==================================================================
# FIGURE 4 -- paired design: the same n subjects under two conditions
# ==================================================================
figure4 <- function() {
  set.seed(7)
  n <- 150
  mu1 <- c(1.2, 0.3); mu2 <- c(-1.0, -0.2)
  Sigma1  <- matrix(c(1.0, 0.3, 0.3, 0.6), 2)
  Sigma2  <- matrix(c(0.6, -0.2, -0.2, 1.0), 2)
  Sigma12 <- matrix(c(0.5, 0.15, 0.1, 0.4), 2)   # = rbind(c(0.5,0.1), c(0.15,0.4))
  joint   <- rbind(cbind(Sigma1, Sigma12), cbind(t(Sigma12), Sigma2))
  stopifnot(min(eigen(joint, symmetric = TRUE)$values) > 0)
  XY <- mvrnorm(n, c(mu1, mu2), joint)
  X1 <- XY[, 1:2]; X2 <- XY[, 3:4]               # row i = same subject

  md  <- colMeans(X1) - colMeans(X2)
  S1  <- cov(X1); S2 <- cov(X2); S12 <- cov(X1, X2)
  ## SSMD keeps the cross-covariance: Sigma* = S1 + S2 - S12 - S12'
  b_ssmd <- orient(solve(S1 + S2 - S12 - t(S12), md), md)
  ## LDA / logistic / SVM treat the two conditions as independent samples
  X <- rbind(X1, X2); y <- c(rep(1, n), rep(0, n))
  beta <- list(
    SSMD     = b_ssmd,
    LDA      = orient(solve(((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2), md), md),
    Logistic = orient(logistic_direction(X, y), md),
    SVM      = orient(svm_direction(X, y), md))

  paired_ssmd <- function(b) { dd <- as.numeric((X1 - X2) %*% b); mean(dd) / sd(dd) }
  pss <- sapply(beta, paired_ssmd)
  ang <- sapply(c("LDA", "Logistic", "SVM"), function(m) angle_deg(beta$SSMD, beta[[m]]))
  cat(sprintf("Figure 4 angles: SSMD-LDA %.1f, SSMD-Logistic %.1f, SSMD-SVM %.1f, LDA-Logistic %.1f\n",
              ang["LDA"], ang["Logistic"], ang["SVM"], angle_deg(beta$LDA, beta$Logistic)))
  cat(sprintf("   paired SSMD_d: SSMD %.2f, SVM %.2f, Logistic %.2f, LDA %.2f\n",
              pss["SSMD"], pss["SVM"], pss["Logistic"], pss["LDA"]))

  ## paired SSMD_d (true, uses S12) and the independence approximation
  ## (ignores S12) as functions of the direction angle
  th <- seq(0, pi, length.out = 721)
  scan <- t(sapply(th, function(t) {
    v  <- c(cos(t), sin(t)); dd <- as.numeric((X1 - X2) %*% v)
    c(paired = abs(mean(dd)) / sd(dd),
      indep  = abs(mean(dd)) / sqrt(as.numeric(t(v) %*% (S1 + S2) %*% v)))
  }))
  dscan <- data.frame(deg = th * 180 / pi, scan)
  i_ind <- which.max(dscan$indep)
  dd_ang <- sapply(beta, dir_deg)
  cat(sprintf("   independence approximation: peak %.2f at %.1f deg (LDA %.1f deg, SSMD %.1f deg)\n",
              dscan$indep[i_ind], dscan$deg[i_ind], dd_ang["LDA"], dd_ang["SSMD"]))

  ## ---- panel (a)
  pairs <- data.frame(x = X1[, 1], y = X1[, 2], xend = X2[, 1], yend = X2[, 2])
  df <- data.frame(x1 = c(X1[, 1], X2[, 1]), x2 = c(X1[, 2], X2[, 2]),
                   g = factor(c(rep("1", n), rep("0", n))))
  arrows <- do.call(rbind, lapply(c("SSMD", "SVM", "Logistic", "LDA"),
                                  function(m) arrow_df(beta[[m]], 3.2, m)))
  pa <- ggplot() +
    geom_segment(data = pairs, aes(x, y, xend = xend, yend = yend),
                 colour = "grey60", linewidth = 0.2, alpha = 0.5) +
    geom_point(data = df, aes(x1, x2, colour = g), alpha = 0.6, size = 1.2) +
    geom_path(data = ellipse_df(colMeans(X1), S1), aes(x, y), colour = COL_G1) +
    geom_path(data = ellipse_df(colMeans(X2), S2), aes(x, y), colour = COL_G2) +
    geom_segment(data = arrows, aes(x, y, xend = xend, yend = yend),
                 colour = METHOD_COLS[arrows$method],
                 linewidth = ifelse(arrows$method == "LDA", 0.7, 1.0),
                 arrow = arrow(length = unit(0.18, "cm"), type = "closed")) +
    annotate("text", x = 3.3, y = 1.55, label = "SSMD (paired)", colour = COL_SSMD,
             hjust = 0, fontface = "bold", size = 3.8) +
    annotate("text", x = 3.3, y = 1.10, label = "SVM", colour = COL_SVM,
             hjust = 0, fontface = "bold", size = 3.8) +
    annotate("text", x = 3.3, y = 0.65, label = "LDA", colour = COL_LDA,
             hjust = 0, fontface = "bold", size = 3.8) +
    annotate("text", x = 3.95, y = 0.65, label = "≈", colour = "black",
             hjust = 0, fontface = "bold", size = 3.8) +
    annotate("text", x = 4.25, y = 0.65, label = "Logistic", colour = COL_LOG,
             hjust = 0, fontface = "bold", size = 3.8) +
    annotate("text", x = 3.3, y = 0.20, label = "(LDA, logistic and SVM ignore pairing)",
             colour = "grey45", hjust = 0, fontface = "italic", size = 3.3) +
    scale_colour_manual(values = GROUP_COLS, breaks = c("1", "0"),
                        labels = c("G1 (condition 1)", "G2 (condition 2)"), name = NULL) +
    coord_cartesian(xlim = c(-5, 8.3), ylim = c(-5, 5)) +
    labs(x = expression(X[1]), y = expression(X[2]),
         title = sprintf("(a) Paired groups (grey lines link the same subject across conditions): non-zero correlation, unequal covariance\nSSMD uses the cross-covariance Σ₁₂ and tilts %.1f° from LDA, %.1f° from logistic and %.1f° from SVM",
                         ang["LDA"], ang["Logistic"], ang["SVM"])) +
    theme(legend.position = c(0.9, 0.1), legend.background = element_rect(colour = "grey80"))

  ## ---- panel (b) paired difference score d = (X1 - X2) beta
  dlist <- lapply(c("SSMD", "SVM", "LDA"), function(m)
    data.frame(d = as.numeric((X1 - X2) %*% beta[[m]]),
               dir = sprintf("%s direction: SSMD_d=%.2f", m, pss[m])))
  dd <- do.call(rbind, dlist)
  dd$dir <- factor(dd$dir, levels = unique(dd$dir))
  dmeans <- aggregate(d ~ dir, dd, mean)
  pb <- ggplot(dd, aes(d, fill = dir)) +
    geom_histogram(aes(y = after_stat(density)), alpha = 0.5, position = "identity", bins = 21) +
    geom_vline(data = dmeans, aes(xintercept = d, colour = dir),
               linetype = c("solid", "dotted", "dashed"), linewidth = 0.8, show.legend = FALSE) +
    geom_vline(xintercept = 0, linetype = "dotted", colour = "grey40") +
    scale_fill_manual(values = c(COL_SSMD, COL_SVM, COL_LDA), name = NULL) +
    scale_colour_manual(values = c(COL_SSMD, COL_SVM, COL_LDA)) +
    labs(x = "paired difference score d", y = "density",
         title = expression("(b) Paired difference score " * bold(d) == (bold(X)[1] - bold(X)[2]) * beta)) +
    theme(legend.position = c(0.25, 0.88), legend.text = element_text(size = 7.5),
          legend.background = element_rect(colour = "grey80"),
          plot.title = element_text(face = "plain", size = 10))

  ## ---- panel (c) paired SSMD_d vs direction angle
  mk <- data.frame(deg = dd_ang[c("SSMD", "SVM", "LDA", "Logistic")],
                   val = pss[c("SSMD", "SVM", "LDA", "Logistic")],
                   what = sprintf("%s (%.1f°)", c("SSMD", "SVM", "LDA", "Logistic"),
                                  dd_ang[c("SSMD", "SVM", "LDA", "Logistic")]))
  mk$what <- factor(mk$what, levels = mk$what)
  curves <- rbind(data.frame(deg = dscan$deg, val = dscan$paired,
                             curve = "paired SSMD_d (uses Σ₁₂)"),
                  data.frame(deg = dscan$deg, val = dscan$indep,
                             curve = "independence approximation (ignores Σ₁₂)"))
  curves$curve <- factor(curves$curve, levels = unique(curves$curve))
  pc <- ggplot() +
    geom_line(data = curves, aes(deg, val, linetype = curve, colour = curve), linewidth = 0.8) +
    geom_vline(data = mk, aes(xintercept = deg), colour = c(COL_SSMD, COL_SVM, COL_LDA, COL_LOG),
               linewidth = 0.6, alpha = 0.85) +
    geom_point(data = mk, aes(deg, val, fill = what), shape = 21, size = 2.6, colour = "white") +
    geom_point(data = dscan[i_ind, ], aes(deg, indep), shape = 25, size = 2.6,
               fill = "grey45", colour = "grey45") +
    annotate("text", x = dscan$deg[i_ind] + 4, y = dscan$indep[i_ind] - 0.15, hjust = 0,
             size = 2.7, colour = "grey35",
             label = sprintf("independence peak (%.2f)", dscan$indep[i_ind])) +
    scale_linetype_manual(values = c("solid", "dashed"), name = NULL) +
    scale_colour_manual(values = c("black", "grey45"), name = NULL) +
    scale_fill_manual(values = c(COL_SSMD, COL_SVM, COL_LDA, COL_LOG), name = NULL) +
    scale_x_continuous(limits = c(0, 180), expand = c(0, 0)) +
    labs(x = "projection-direction angle (degrees)", y = expression("paired SSMD"[d]),
         title = expression("(c) Only the SSMD direction maximizes the paired SSMD"[d])) +
    guides(linetype = guide_legend(order = 1, nrow = 1), colour = guide_legend(order = 1, nrow = 1),
           fill = guide_legend(order = 2, nrow = 1)) +
    theme(legend.position = "bottom", legend.box = "vertical", legend.text = element_text(size = 7),
          legend.margin = margin(0, 0, 0, 0), legend.spacing.y = unit(0, "pt"),
          plot.title = element_text(face = "plain", size = 10))

  g <- arrangeGrob(pa, pb, pc, layout_matrix = rbind(c(1, 1), c(2, 3)), heights = c(1.3, 1))
  save_png(file.path(OUT, "figure4_R.png"), g, width = 13, height = 9.2)
}

figure2()
figure3()
figure4()
invisible(dev.off())
cat("saved figure2_R.png, figure3_R.png, figure4_R.png to", normalizePath(OUT), "\n")
