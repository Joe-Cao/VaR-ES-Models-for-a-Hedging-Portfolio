# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 08:41:59 2025

@author: caoxi
"""

# backtesting.py
from __future__ import annotations
import numpy as np
import pandas as pd

from var_models import (
    historical_var_es,
    parametric_var_es,
    mc_heston_deltahedge_portfolio_var_es_for_day,
)


def rolling_var_backtest(
    legs_df,
    dates,
    losses,
    alpha: float = 0.95,
    window: int = 20,
    method: str = "hist",
    mc_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Compute rolling VaR/ES on a loss series and flag violations.

    Parameters
    ----------
    legs_df: Daily PnL information for each leg/asset in the delta-hedge.
    dates: 1D array-like of datetime64.
    losses: 1D array-like of float, L_t = -PnL_t (positive means loss).
    alpha: VaR confidence level, e.g., 0.95 or 0.99.
    window: Rolling window length (in days), e.g., 20.
    method: "hist" / "param" / "mc".
    mc_samples: Number of MC bootstrap samples (used only when method="mc").
    rng: Optional numpy.random.Generator.

    Returns
    -------
    DataFrame with columns: date, loss, VaR, ES, violation.
    """
    dates = pd.to_datetime(dates)
    losses = np.asarray(losses, dtype=float)
    n = len(losses)

    rows: list[dict] = []
    if method == "mc":
        t = 0
        for date, day_df in legs_df.groupby("date"):
            var_t, es_t = mc_heston_deltahedge_portfolio_var_es_for_day(
                    day_df,
                    alpha=alpha,
                    n_paths=1000,
                    n_steps=1,
                    dt_year=1.0/252.0,
                )
            loss_t = losses[t]
            violation = bool(loss_t > var_t)
            rows.append(
                {
                    "date": dates[t],
                    "loss": loss_t,
                    "VaR": var_t,
                    "ES": es_t,
                    "violation": violation,
                }
            )
            t += 1
        return pd.DataFrame(rows)
    
    else:
        for t in range(window, n):
            window_losses = losses[t - window: t]
            if method == "hist":
                var_t, es_t = historical_var_es(window_losses, alpha)
            elif method == "param":
                var_t, es_t = parametric_var_es(window_losses, alpha)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            loss_t = losses[t]
            violation = bool(loss_t > var_t)
            rows.append(
                {
                    "date": dates[t],
                    "loss": loss_t,
                    "VaR": var_t,
                    "ES": es_t,
                    "violation": violation,
                }
            )
        return pd.DataFrame(rows)