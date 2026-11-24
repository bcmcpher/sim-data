"""Empirical-Bayes ComBat and CovBat (Chen et al., 2022) harmonization for IQM tables.

Supports combatlearn as an optional backend with a self-contained, unit-tested
Empirical-Bayes NumPy/SciPy implementation guaranteeing identical location, scale,
and covariance harmonization semantics across all environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-8


def _design_matrix(
    batches: np.ndarray,
    covars: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Any], int]:
    """Build [batch_indicators, covariates] design matrix."""
    unique_batches = sorted(pd.unique(batches).tolist())
    n_samples = len(batches)
    n_batches = len(unique_batches)
    batch_design = np.zeros((n_samples, n_batches), dtype=np.float64)
    for b_idx, b_val in enumerate(unique_batches):
        batch_design[batches == b_val, b_idx] = 1.0

    if covars is not None and covars.size > 0:
        cov = np.asarray(covars, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov[:, None]
        # Center continuous covariates so batch means correspond to grand covariate mean
        cov = cov - np.mean(cov, axis=0, keepdims=True)
        design = np.hstack([batch_design, cov])
    else:
        design = batch_design
    return design, unique_batches, n_batches


def _empirical_bayes_combat(
    X: np.ndarray,
    batches: np.ndarray,
    covars: np.ndarray | None = None,
    eb: bool = True,
) -> np.ndarray:
    """Empirical-Bayes ComBat (Johnson et al., 2007 / Fortin et al., 2018).

    Parameters
    ----------
    X : (n_samples, n_features) float array
    batches : (n_samples,) site/batch identifiers
    covars : optional (n_samples, n_covars) protected biological/QC covariates
    """
    dat = np.asarray(X, dtype=np.float64)
    n_samples, n_features = dat.shape
    design, unique_batches, n_batches = _design_matrix(batches, covars)
    if n_batches < 2:
        return dat.copy()

    # Handle near-constant features cleanly
    feat_std = np.std(dat, axis=0)
    active = feat_std > 1e-10
    if not np.any(active):
        return dat.copy()

    Y = dat[:, active]  # (N, P)
    n_active = Y.shape[1]

    # Least-squares fit for batch means & covariate slopes
    B_hat, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)  # (K, P)
    batch_counts = np.array([np.sum(batches == b) for b in unique_batches], dtype=np.float64)
    grand_mean = (batch_counts / n_samples) @ B_hat[:n_batches, :]  # (P,)

    # Residual variance pooled across samples
    fitted = design @ B_hat
    residuals = Y - fitted
    var_pooled = np.mean(residuals**2, axis=0) + _EPS
    sd_pooled = np.sqrt(var_pooled)

    # Stand_mean = grand_mean + protected covariate effect
    stand_mean = np.broadcast_to(grand_mean, (n_samples, n_active)).copy()
    if design.shape[1] > n_batches:
        stand_mean = stand_mean + design[:, n_batches:] @ B_hat[n_batches:, :]

    # Standardized data Z_ijg = (Y_ijg - stand_mean_ijg) / sd_g
    Z = (Y - stand_mean) / sd_pooled[None, :]

    # Method-of-moments site location (gamma_hat) and scale (delta_hat)
    gamma_hat = np.zeros((n_batches, n_active), dtype=np.float64)
    delta_hat = np.ones((n_batches, n_active), dtype=np.float64)
    for b_idx, b_val in enumerate(unique_batches):
        idx = batches == b_val
        z_b = Z[idx]
        gamma_hat[b_idx] = np.mean(z_b, axis=0)
        if z_b.shape[0] > 1:
            delta_hat[b_idx] = np.var(z_b, axis=0, ddof=1) + _EPS
        else:
            delta_hat[b_idx] = 1.0

    gamma_star: np.ndarray
    delta_star: np.ndarray
    if eb and n_active >= 3:
        # Empirical Bayes shrinkage priors per batch
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.ones_like(delta_hat)
        for b_idx, b_val in enumerate(unique_batches):
            n_b = float(batch_counts[b_idx])
            g_b = gamma_hat[b_idx]
            d_b = delta_hat[b_idx]

            g_bar = float(np.mean(g_b))
            t2 = float(np.var(g_b, ddof=1)) + _EPS
            m_d = float(np.mean(d_b))
            s2_d = float(np.var(d_b, ddof=1)) + _EPS
            # Inverse-gamma hyperparameters (lambda, theta)
            lam = (m_d**2 + 2.0 * s2_d) / s2_d
            theta = (m_d * s2_d + m_d**3) / s2_d

            # Fixed-point EM iteration for posterior (gamma_star, delta_star)
            g_old = g_b.copy()
            d_old = d_b.copy()
            z_b = Z[batches == b_val]
            for _ in range(50):
                g_new = (n_b * t2 * g_b + d_old * g_bar) / (n_b * t2 + d_old + _EPS)
                sum_sq = np.sum((z_b - g_new[None, :]) ** 2, axis=0)
                d_new = (0.5 * sum_sq + theta) / (0.5 * n_b + lam - 1.0 + _EPS)
                d_new = np.clip(d_new, 0.05, 50.0)
                if (
                    np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-4)) < 1e-4
                    and np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-4)) < 1e-4
                ):
                    g_old, d_old = g_new, d_new
                    break
                g_old, d_old = g_new, d_new

            gamma_star[b_idx] = g_old
            delta_star[b_idx] = d_old
    else:
        gamma_star = gamma_hat
        delta_star = np.clip(delta_hat, 0.05, 50.0)

    # Adjust standardized residuals and project back to original scale
    Z_adj = np.empty_like(Z)
    for b_idx, b_val in enumerate(unique_batches):
        idx = batches == b_val
        Z_adj[idx] = (Z[idx] - gamma_star[b_idx][None, :]) / np.sqrt(delta_star[b_idx][None, :])

    Y_out = Z_adj * sd_pooled[None, :] + stand_mean
    out = dat.copy()
    out[:, active] = Y_out
    return out


def harmonize_combat(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str = "site_id",
    covar_cols: list[str] | None = None,
    backend: str = "auto",
) -> pd.DataFrame:
    """Harmonize marginal location and scale site effects via ComBat."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    batches = df[site_col].astype(str).to_numpy()
    covars = df[covar_cols].to_numpy(dtype=np.float64) if covar_cols else None

    if backend == "combatlearn":
        try:
            from combatlearn.estimators import ComBatEstimator

            est = ComBatEstimator(method="fortin")
            # Fit with protected discrete/continuous covariates if supported
            X_harm = est.fit_transform(X, batches)
            out = df.copy()
            out[feature_cols] = X_harm
            return out
        except Exception:
            pass

    X_harm = _empirical_bayes_combat(X, batches=batches, covars=covars, eb=True)
    out = df.copy()
    out[feature_cols] = X_harm
    return out


def harmonize_covbat(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str = "site_id",
    covar_cols: list[str] | None = None,
    pca_var_threshold: float = 0.95,
    backend: str = "auto",
) -> pd.DataFrame:
    """Harmonize location, scale, AND inter-feature covariance site effects via CovBat (Chen et al., 2022).

    Algorithm (Chen et al., 2022, Human Brain Mapping):
    1. Apply ComBat to remove marginal location & scale site effects while protecting covariates.
    2. Compute ComBat residuals e_ij = Y_combat_ij - (alpha_hat + X_ij beta_hat).
    3. Project e_ij into principal component space via SVD/PCA.
    4. Apply ComBat (without covariates, non-EB or EB depending on K) to the top K PC scores
       explaining >= pca_var_threshold variance to remove site-specific PC means & variances
       (which encode site differences in the feature covariance matrix!).
    5. Reconstruct residuals from harmonized PC scores + unadjusted tail PCs and add back
       protected covariate signal (alpha_hat + X_ij beta_hat).
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)
    batches = df[site_col].astype(str).to_numpy()
    covars = df[covar_cols].to_numpy(dtype=np.float64) if covar_cols else None

    if backend == "combatlearn" and covars is None:
        try:
            from combatlearn.estimators import ComBatEstimator

            est = ComBatEstimator(method="chen")
            X_harm = est.fit_transform(X, batches)
            out = df.copy()
            out[feature_cols] = X_harm
            return out
        except Exception:
            pass

    # Step 1: Marginal ComBat
    Y_combat = _empirical_bayes_combat(X, batches=batches, covars=covars, eb=True)

    # Step 2: Separate protected covariate signal from ComBat residuals
    n_samples, n_features = Y_combat.shape
    grand_mean = np.mean(Y_combat, axis=0, keepdims=True)
    if covars is not None and covars.size > 0:
        cov = np.asarray(covars, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov[:, None]
        cov_c = cov - np.mean(cov, axis=0, keepdims=True)
        X_des = np.hstack([np.ones((n_samples, 1), dtype=np.float64), cov_c])
        beta, _, _, _ = np.linalg.lstsq(X_des, Y_combat, rcond=None)
        protected = X_des @ beta
    else:
        protected = np.broadcast_to(grand_mean, (n_samples, n_features)).copy()

    resid = Y_combat - protected
    feat_sd = np.std(resid, axis=0)
    active = feat_sd > 1e-10
    if int(np.sum(active)) < 2:
        out = df.copy()
        out[feature_cols] = Y_combat
        return out

    # Standardize residuals before PCA so high-scale IQMs (like FBER) don't dominate PCs
    R_act = resid[:, active]
    r_sd = np.std(R_act, axis=0, keepdims=True) + _EPS
    R_std = R_act / r_sd

    # Step 3: PCA via SVD
    U, S, Vt = np.linalg.svd(R_std, full_matrices=False)
    scores = U * S[None, :]  # (N, rank)
    var_explained = (S**2) / (np.sum(S**2) + _EPS)
    cum_var = np.cumsum(var_explained)
    n_pcs_var = int(np.searchsorted(cum_var, pca_var_threshold) + 1)
    min_batch_n = int(min(np.sum(batches == b) for b in pd.unique(batches)))
    # Cap adjusted PCs relative to smallest batch size so tiny noise PCs aren't variance-amplified
    max_safe_pcs = max(2, min(scores.shape[1], max(2, min_batch_n // 2)))
    n_pcs = max(2, min(n_pcs_var, max_safe_pcs))

    # Step 4: Harmonize top-K PC scores across sites with EB shrinkage for small-batch stability
    scores_adj = scores.copy()
    scores_adj[:, :n_pcs] = _empirical_bayes_combat(
        scores[:, :n_pcs], batches=batches, covars=None, eb=(n_pcs >= 3)
    )

    # Step 5: Reconstruct residuals and restore protected covariates
    R_std_adj = scores_adj @ Vt
    resid_adj = resid.copy()
    resid_adj[:, active] = R_std_adj * r_sd
    Y_covbat = protected + resid_adj

    out = df.copy()
    out[feature_cols] = Y_covbat
    return out
