# ==================================================================
# Monte Carlo summary for Simulation 2 (Figure 3): replicate datasets
# from the same design, showing that the SSMD-LDA divergence is
# structural (population-level) rather than a property of one seed.
#
# Usage:   Rscript Simulation2.montecarlo.R [replicates]   (default 1000)
# Requires: MASS, e1071, pROC
# ==================================================================
suppressPackageStartupMessages({ library(MASS); library(e1071); library(pROC) })
args <- commandArgs(trailingOnly = TRUE)
R <- if (length(args) >= 1) as.integer(args[1]) else 1000

mu1 <- c(1.5, 0); mu2 <- c(-1.5, 0); m <- mu1 - mu2
S1 <- matrix(c(0.5, -0.8, -0.8, 3.0), 2); S2 <- matrix(c(4.0, 1.2, 1.2, 1.0), 2)
n1 <- 80; n2 <- 320; pi1 <- 0.2; pi2 <- 0.8

orient <- function(b, md) { b <- as.numeric(b) / sqrt(sum(b^2)); if (sum(b * md) < 0) -b else b }
ang  <- function(a, b) acos(min(1, abs(sum(a * b)))) * 180 / pi
sdeg <- function(b) atan2(b[2], b[1]) * 180 / pi
pop_ssmd <- function(b) sum(b * m) / sqrt(as.numeric(t(b) %*% (S1 + S2) %*% b))  # true SSMD
pop_err  <- function(b, c)                                                        # true error of b'x > c
  pi1 * pnorm((c - sum(b * mu1)) / sqrt(as.numeric(t(b) %*% S1 %*% b))) +
  pi2 * pnorm((c - sum(b * mu2)) / sqrt(as.numeric(t(b) %*% S2 %*% b)), lower.tail = FALSE)
logistic_direction <- function(X, y) as.numeric(coef(suppressWarnings(glm(y ~ X, family = binomial)))[-1])
svm_direction <- function(X, y, cost = 1) {
  Xsc <- scale(X)
  fit <- svm(Xsc, factor(y), kernel = "linear", cost = cost, scale = FALSE)
  as.numeric(t(fit$coefs) %*% fit$SV) / attr(Xsc, "scaled:scale")
}

## population (true-parameter) directions
set.seed(1)
Xb <- rbind(mvrnorm(40000, mu1, S1), mvrnorm(160000, mu2, S2)); yb <- c(rep(1, 40000), rep(0, 160000))
pop <- list(SSMD = orient(solve(S1 + S2, m), m),
            LDA  = orient(solve(pi1 * S1 + pi2 * S2, m), m),
            Logistic = orient(logistic_direction(Xb, yb), m))
cat("Population (true-parameter) directions:\n")
for (k in names(pop))
  cat(sprintf("  %-9s angle %+6.1f deg  SSMD(beta) %.3f  AUROC %.3f\n",
              k, sdeg(pop[[k]]), pop_ssmd(pop[[k]]), pnorm(pop_ssmd(pop[[k]]))))
bs <- pop$SSMD
opt <- optimize(function(c) pop_err(bs, c), c(sum(bs * mu2), sum(bs * mu1)))
cat(sprintf("  SSMD direction: midpoint error %.3f, Bayes error %.3f\n",
            pop_err(bs, (sum(bs * mu1) + sum(bs * mu2)) / 2), opt$objective))

## replicates
set.seed(2026)
D <- t(replicate(R, {
  X1 <- mvrnorm(n1, mu1, S1); X2 <- mvrnorm(n2, mu2, S2)
  X <- rbind(X1, X2); y <- c(rep(1, n1), rep(0, n2))
  md <- colMeans(X1) - colMeans(X2); s1 <- cov(X1); s2 <- cov(X2)
  B <- list(orient(solve(s1 + s2, md), md),
            orient(solve(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2), md), md),
            orient(logistic_direction(X, y), md),
            orient(svm_direction(X, y), md))
  s <- as.numeric(X %*% B[[1]])
  a1 <- mean(s[y == 1]); a2 <- mean(s[y == 0]); v1 <- var(s[y == 1]); v2 <- var(s[y == 0])
  qa <- 1 / (2 * v1) - 1 / (2 * v2); qb <- -a1 / v1 + a2 / v2
  qd <- a1^2 / (2 * v1) - a2^2 / (2 * v2) - log((pi1 / sqrt(v1)) / (pi2 / sqrt(v2)))
  rt <- Re(polyroot(c(qd, qb, qa))); bayes <- rt[rt >= a2 & rt <= a1][1]
  youden <- as.numeric(unlist(coords(roc(y, s, quiet = TRUE, direction = "<"), "best",
                                     best.method = "youden", ret = "threshold")))[1]
  c(ang(B[[1]], B[[2]]), ang(B[[1]], B[[3]]), ang(B[[1]], B[[4]]),
    sapply(B, function(b) pnorm(pop_ssmd(b))),
    pop_err(B[[1]], (a1 + a2) / 2), pop_err(B[[1]], youden), pop_err(B[[1]], bayes))
}))
lab <- c("angle SSMD-LDA", "angle SSMD-Logistic", "angle SSMD-SVM",
         "true AUROC, SSMD dir", "true AUROC, LDA dir", "true AUROC, Logistic dir", "true AUROC, SVM dir",
         "true error, midpoint", "true error, Youden", "true error, Bayes")
cat(sprintf("\n%d replicates: median [5th, 95th percentile]\n", R))
for (j in seq_along(lab)) {
  q <- quantile(D[, j], c(0.5, 0.05, 0.95))
  cat(sprintf("  %-26s %.3f [%.3f, %.3f]\n", lab[j], q[1], q[2], q[3]))
}
cat(sprintf("  P(SSMD dir has highest true AUROC) = %.3f\n", mean(D[, 4] >= apply(D[, 5:7], 1, max))))
cat(sprintf("  P(true AUROC SSMD > LDA) = %.3f;  P(> SVM) = %.3f;  P(> Logistic) = %.3f\n",
            mean(D[, 4] > D[, 5]), mean(D[, 4] > D[, 7]), mean(D[, 4] > D[, 6])))
cat(sprintf("  P(Bayes < Youden < midpoint error) = %.3f\n", mean(D[, 10] < D[, 9] & D[, 9] < D[, 8])))
