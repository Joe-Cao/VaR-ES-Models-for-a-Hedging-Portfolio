# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 04:29:54 2026

@author: caoxi
"""

# es_tail_analysis.py
"""
ES tail-risk analysis utilities.

Core function:
    es_tail_summary(results_dict, alpha, tail_alpha=None)

Where:
- results_dict: dict[str, pd.DataFrame]
    Each DataFrame comes from rolling_var_backtest and must contain columns:
        'date', 'loss', 'ES'
- alpha: the confidence level used when computing VaR/ES (e.g., 0.95 or 0.99)
- tail_alpha: the quantile level used to define the “worst (1 - tail_alpha)% losses”.
    Default = alpha, i.e. the tail probability is consistent with VaR/ES.
"""


#from __future__ import annotations

import numpy as np
import pandas as pd


def es_tail_summary(
    results_dict: dict[str, pd.DataFrame],
    alpha: float,
    tail_alpha: float | None = None,
) -> pd.DataFrame:
    """
    Compare tail risk across different VaR/ES models.

    Parameters
    ----------
    results_dict:
        Keys are method names (e.g. "hist", "param", "mc").
        Values are DataFrames returned by rolling_var_backtest and must contain
        at least the columns: 'loss', 'ES'.

    alpha:
        Confidence level for VaR / ES (e.g. 0.95, 0.99).

    tail_alpha:
        Quantile level used to define the “worst (1 - tail_alpha)% losses”.
        If None, it is set to alpha, meaning the same tail probability is used
        as for VaR / ES.

    Returns
    -------
    A DataFrame with one row per method, containing:
        - method_key: the key in results_dict
        - n_obs: backtest sample size
        - alpha_for_ES: alpha used for ES
        - tail_alpha: alpha used to define the tail set
        - empirical_tail_quantile: tail_alpha quantile of realized losses
        - n_tail: number of observations in the tail set
        - empirical_ES_tail: average realized loss in the tail (empirical ES)
        - mean_pred_ES_all: average predicted ES over the full sample
        - mean_pred_ES_tail: average predicted ES over tail days only
        - bias_tail: mean_pred_ES_tail - empirical_ES_tail
        - rel_bias_tail: (mean_pred_ES_tail / empirical_ES_tail - 1)
    """
    if not results_dict:
        return pd.DataFrame()

    if tail_alpha is None:
        tail_alpha = alpha

    rows: list[dict] = []

    for key, df in results_dict.items():
        # Defensive copy + type casting
        df = df.copy()
        if "loss" not in df.columns or "ES" not in df.columns:
            raise ValueError(
                f"DataFrame for method '{key}' must contain 'loss' and 'ES' columns."
            )

        df["loss"] = df["loss"].astype(float)
        df["ES"] = df["ES"].astype(float)

        losses = df["loss"].values
        n_obs = len(losses)

        if n_obs == 0:
            rows.append(
                {
                    "method_key": key,
                    "n_obs": 0,
                    "alpha_for_ES": alpha,
                    "tail_alpha": tail_alpha,
                    "empirical_tail_quantile": np.nan,
                    "n_tail": 0,
                    "empirical_ES_tail": np.nan,
                    "mean_pred_ES_all": np.nan,
                    "mean_pred_ES_tail": np.nan,
                    "bias_tail": np.nan,
                    "rel_bias_tail": np.nan,
                }
            )
            continue

        # Define the “tail” using realized losses:
        #   worst (1 - tail_alpha)% losses
        q = np.quantile(losses, tail_alpha)
        tail_mask = losses >= q
        n_tail = int(tail_mask.sum())

        empirical_ES_tail = (
            float(losses[tail_mask].mean()) if n_tail > 0 else np.nan
        )

        # Model-implied ES (predicted values)
        mean_pred_ES_all = float(df["ES"].mean())
        mean_pred_ES_tail = (
            float(df.loc[tail_mask, "ES"].mean()) if n_tail > 0 else np.nan
        )

        if n_tail > 0 and empirical_ES_tail != 0 and np.isfinite(empirical_ES_tail):
            bias_tail = mean_pred_ES_tail - empirical_ES_tail
            rel_bias_tail = mean_pred_ES_tail / empirical_ES_tail - 1.0
        else:
            bias_tail = np.nan
            rel_bias_tail = np.nan

        row = {
            "method_key": key,
            "n_obs": n_obs,
            "alpha_for_ES": alpha,
            "tail_alpha": tail_alpha,
            "empirical_tail_quantile": q,
            "n_tail": n_tail,
            "empirical_ES_tail": empirical_ES_tail,
            "mean_pred_ES_all": mean_pred_ES_all,
            "mean_pred_ES_tail": mean_pred_ES_tail,
            "bias_tail": bias_tail,
            "rel_bias_tail": rel_bias_tail,
        }
        rows.append(row)

    return pd.DataFrame(rows)
