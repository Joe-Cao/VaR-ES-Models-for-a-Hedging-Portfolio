# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 22:49:37 2025

@author: caoxi
"""

# var_models.py
import numpy as np
import pandas as pd
from scipy.stats import norm
from calibrate_heston_bs import *
from delta_hedging_backtest_tailored import *


def historical_var_es(losses, alpha: float = 0.95):
    """
    Historical VaR/ES
    losses: 1D array-like, loss samples (positive = losing money)
    VaR_alpha = the sample's alpha quantile
    ES_alpha  = the average loss of those sample points that are >= VaR
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return np.nan, np.nan

    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    es = tail.mean() if tail.size > 0 else var
    return var, es


def parametric_var_es(losses, alpha: float = 0.95):
    """
    Parametric VaR/ES (normality assumption)
    L ~ N(mu, sigma^2)
      VaR_alpha = mu + z_alpha * sigma
      ES_alpha  = mu + sigma * phi(z_alpha) / (1 - alpha)
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return np.nan, np.nan

    mu = losses.mean()
    sigma = losses.std(ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return mu, mu  

    z = norm.ppf(alpha)
    var = mu + z * sigma
    pdf = norm.pdf(z)
    es = mu + sigma * pdf / (1.0 - alpha)
    return var, es

# ---------------------------------------------------------------------
# Below is the core MC VaR function for Heston + delta-hedge
# ---------------------------------------------------------------------


def mc_heston_deltahedge_portfolio_var_es_for_day(
    day_df: "pd.DataFrame",
    alpha: float = 0.95,
    n_paths: int = 1000,
    n_steps: int = 1,
    dt_year: float = 1.0 / 252.0,
    rng: np.random.Generator | None = None,
):
    """
    Compute Heston MC delta-hedged portfolio VaR/ES for a basket of options on a given quote date.

    Parameters
    ----------
    day_df: The sub-table of all options on that date; one row per option.
            It must contain at least the following columns:
        - 'S_i'      : spot price on that date
        - 'strike'   : K
        - 'T_i'      : remaining time to maturity (in years)
        - 'cp'       : 'C' or 'P'
        - 'r', 'q'   : risk-free rate / dividend yield (same for the day)
        - 'kappa_i','theta_i','sigma_i','rho_i','v0_i': Heston parameters (same for the day)

      If you have position size, you can add a column such as 'qty'; here each contract defaults to notional 1.

    Returns
    -------
        var_port, es_port, losses_port
    """
    if rng is None:
        rng = np.random.default_rng()

    if day_df.empty:
        return np.nan, np.nan, np.array([])

    # Use the first row of that day to extract "global" parameters (same across all options that day)
    row0 = day_df.iloc[0]
    S0 = float(row0["S_i"])
    r = float(row0["r"])
    q = float(row0["q"])
    kappa = float(row0["kappa_i"])
    theta = float(row0["theta_i"])
    sigma_v = float(row0["sigma_i"])
    rho = float(row0["rho_i"])
    v0 = float(row0["v0_i"])

    # 1) Construct the Heston parameter object
    params0 = HestonParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma_v),
        rho=float(rho),
        v0=float(v0),
    )

    # 2) Simulate the underlying path S_t -> S_{t+dt} (shared by all options)
    S = np.full(n_paths, S0, dtype=float)
    v = np.full(n_paths, v0, dtype=float)

    dt = dt_year / n_steps
    sqrt_dt = np.sqrt(dt)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2_indep = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2_indep

        v = v + kappa * (theta - v) * dt + sigma_v * np.sqrt(np.maximum(v, 0.0)) * sqrt_dt * z2
        v = np.maximum(v, 0.0)

        S = S * np.exp((r - q - 0.5 * v) * dt + np.sqrt(v * dt) * z1)

    # 3) Initialize the portfolio PnL on each path
    pnl_portfolio = np.zeros(n_paths, dtype=float)

    # 4) For each option i of the day, accumulate its PnL on each path
    for _, opt in day_df.iterrows():
        K = float(opt["strike"])
        tau = float(opt["T_i"])
        cp = str(opt["cp"]).upper()
        qty = float(opt.get("qty", 1.0))  # use the position column if present; otherwise default to 1

        # Starting price & Delta
        C0 = heston_price(S0, K, tau, r, q, params0, cp)
        delta0 = heston_delta_fd(S0, K, tau, r, q, params0, cp)

        # Ending price
        tau_next = max(tau - dt_year, 0.0)
        C_next = np.empty_like(S)
        if tau_next <= 0.0:
            # At maturity: intrinsic value
            if cp == "C":
                C_next = np.maximum(S - K, 0.0)
            else:
                C_next = np.maximum(K - S, 0.0)
        else:
            for i in range(n_paths):
                C_next[i] = heston_price(S[i], K, tau_next, r, q, params0, cp)

        # This option's no-bond delta-hedge PnL on each path
        pnl_i = (C_next - C0) - delta0 * (S - S0)

        # Multiply by quantity and add to the portfolio
        pnl_portfolio += qty * pnl_i

    # 5) Portfolio losses and VaR/ES
    losses_port = -pnl_portfolio
    var_port = np.quantile(losses_port, alpha)
    tail = losses_port[losses_port >= var_port]
    es_port = tail.mean() if tail.size > 0 else var_port

    return var_port, es_port