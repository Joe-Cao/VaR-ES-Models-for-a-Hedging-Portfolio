# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:44:10 2025

@author: caoxi
"""

# data_utils.py
import pandas as pd


def load_pnl(csv_path: str) -> pd.DataFrame:
    """
    Read daily PnL data from a delta-hedge backtest.
    The CSV is expected to contain at least the columns:
      date, PnL_total, TH, UM_delta_hedged, SM, ME, TC, cash_pnl

    Returns a DataFrame with:
      date, PnL_total, loss ( = -PnL_total ), and all available component columns.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Define loss as a positive number: loss_t = -PnL_total_t
    df["loss"] = -df["PnL_total"]

    cols = ["date", "PnL_total", "loss",
            "TH", "UM_delta_hedged", "SM", "ME", "TC", "cash_pnl"]
    # Avoid KeyError if some component columns are missing
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def load_heston_params(csv_path: str) -> pd.DataFrame:
    """
    Read Heston calibration parameters.
    Expected columns include: quote_date, kappa, theta, sigma, rho, v0,
    rmse_price, mae_price, n_points, etc.
    """
    df = pd.read_csv(csv_path, parse_dates=["quote_date"])
    df = df.sort_values("quote_date").reset_index(drop=True)
    return df


def merge_params_pnl(params: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the Heston-parameter table and the PnL table by date for downstream analysis.
    """
    merged = pnl.merge(
        params,
        left_on="date",
        right_on="quote_date",
        how="left",
        suffixes=("", "_heston")
    )
    return merged
