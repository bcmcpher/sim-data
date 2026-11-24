"""ComBat & CovBat (Chen et al., 2022) harmonization backed by `combatlearn`.

Delegates location/scale Empirical-Bayes ComBat (`method="fortin"`) and covariance
harmonization CovBat (`method="chen"`) directly to `combatlearn.sklearn_api.ComBat`.
Standardizes heterogeneous-unit IQM features (e.g., EFC in [0,1] vs FBER in [1,1e4])
before CovBat PCA and inverse-transforms back to original IQM units afterward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _run_combatlearn(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str,
    covar_cols: list[str] | None,
    method: str,
    covbat_cov_thresh: float = 0.90,
) -> pd.DataFrame:
    try:
        from combatlearn.sklearn_api import ComBat
    except ImportError as exc:
        raise ImportError(
            "Harmonization requires `combatlearn`. Install with: "
            "`uv sync --extra cpu --extra harmonization`"
        ) from exc

    X = df[feature_cols].to_numpy(dtype=np.float64)
    batches = df[site_col].astype(str).to_numpy()
    if len(pd.unique(batches)) < 2:
        return df.copy()

    feat_std = np.std(X, axis=0)
    active = feat_std > 1e-10
    if not np.any(active):
        return df.copy()

    # Scale heterogeneous-unit IQMs to unit variance so CovBat PCA weights all IQMs equally
    scaler = StandardScaler()
    X_act_std = scaler.fit_transform(X[:, active])
    covars = (
        df[covar_cols].to_numpy(dtype=np.float64)
        if (covar_cols is not None and len(covar_cols) > 0)
        else None
    )

    eff_thresh: float | int = covbat_cov_thresh
    if method == "chen":
        min_batch_n = int(min(np.sum(batches == b) for b in pd.unique(batches)))
        max_safe_pcs = max(2, min(X_act_std.shape[1], max(2, min_batch_n // 3)))
        if isinstance(covbat_cov_thresh, float) and X_act_std.shape[1] > max_safe_pcs:
            # Cap adjusted PCs on small batches so low-variance noise PCs aren't inflated
            eff_thresh = max_safe_pcs

    est = ComBat(
        batch=batches,
        continuous_covariates=covars,
        method=method,
        parametric=True,
        covbat_cov_thresh=eff_thresh,
    )
    X_harm_std = np.asarray(est.fit_transform(X_act_std), dtype=np.float64)
    X_harm_act = scaler.inverse_transform(X_harm_std)

    X_out = X.copy()
    X_out[:, active] = X_harm_act
    out = df.copy()
    out[feature_cols] = X_out
    return out


def harmonize_combat(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str = "site_id",
    covar_cols: list[str] | None = None,
    backend: str = "auto",
) -> pd.DataFrame:
    """Harmonize marginal site location and scale effects via `combatlearn` (`method='fortin'`)."""
    _ = backend
    return _run_combatlearn(
        df=df,
        feature_cols=feature_cols,
        site_col=site_col,
        covar_cols=covar_cols,
        method="fortin",
    )


def harmonize_covbat(
    df: pd.DataFrame,
    feature_cols: list[str],
    site_col: str = "site_id",
    covar_cols: list[str] | None = None,
    pca_var_threshold: float = 0.90,
    backend: str = "auto",
) -> pd.DataFrame:
    """Harmonize location, scale, and covariance site effects via `combatlearn` CovBat (`method='chen'`)."""
    _ = backend
    return _run_combatlearn(
        df=df,
        feature_cols=feature_cols,
        site_col=site_col,
        covar_cols=covar_cols,
        method="chen",
        covbat_cov_thresh=pca_var_threshold,
    )
