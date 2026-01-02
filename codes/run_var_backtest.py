# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 10:10:35 2025

@author: caoxi
"""

# run_var_backtest.py
"""
Usage example (from terminal):
    python run_var_backtest.py

You can modify the following in main():
  - pnl_csv_path       -> path to your “second table” (daily PnL)
  - params_csv_path    -> path to your “first table” (Heston params)
  - alpha, window      -> VaR confidence & rolling window length
"""

#from __future__ import annotations
import numpy as np
import pandas as pd

from data_utils import load_pnl, load_heston_params, merge_params_pnl
from backtesting import rolling_var_backtest
from stats_tests import christoffersen_cc_test
from es_tail_analysis import es_tail_summary


def main():
    # === 1. Path settings (replace with your own) ==========================
    pnl_csv_path = r'C:\Users\caoxi\python练习\Var model\daily_df.csv'     
    params_csv_path = r'C:\Users\caoxi\python练习\Var model\calib_csv.csv' 

    # === 2. Load data ======================================================
    pnl_df = load_pnl(pnl_csv_path)
    params_df = load_heston_params(params_csv_path)
    merged = merge_params_pnl(params_df, pnl_df)

    # Loss series & dates
    dates = merged["date"]
    losses = merged["loss"].values

    # VaR confidence & rolling window
    alpha = 0.95       # start with 95% VaR; you can also run 99% later
    window = 20        # 20-day rolling window

    print(f"Using alpha={alpha}, window={window}, "
          f"n_obs={len(losses)}, n_backtest={len(losses)-window}")

    # === 3. Three VaR backtests ===========================================
    methods = {
        "hist": "Historical",
        "param": "ParametricNormal",
        "mc": "MC_Bootstrap",
    }

    rng = np.random.default_rng(12345)

    results = {}
    stats_rows = []   # collect per-method statistics

    for m_key, m_name in methods.items():
        df_bt = rolling_var_backtest(
            dates=dates,
            losses=losses,
            alpha=alpha,
            window=window,
            method=m_key,
            mc_samples=10_000,
            rng=rng,
        )
        results[m_key] = df_bt

        # Save daily backtest results to CSV so you can plot them yourself
        out_csv = f"var_backtest_{m_key}.csv"
        df_bt.to_csv(out_csv, index=False)
        print(f"\n=== {m_name} VaR Backtest ===")
        print(f"Saved detailed backtest to: {out_csv}")

        # === 4. Kupiec + Christoffersen tests ==============================
        stats = christoffersen_cc_test(df_bt["violation"].values, alpha=alpha)

        # Print to terminal
        print(f"T        = {stats['T']}")
        print(f"N_viol   = {stats['N1']}")
        print(f"pi_hat   = {stats['pi_hat']:.4f} (theoretical p = {1-alpha:.4f})")
        print(f"LR_uc    = {stats['LR_uc']:.4f},  p_uc  = {stats['p_value']:.4f}")
        print(f"LR_ind   = {stats['LR_ind']:.4f}, p_ind = {stats['p_value_ind']:.4f}")
        print(f"LR_cc    = {stats['LR_cc']:.4f},  p_cc  = {stats['p_value_cc']:.4f}")

        # Gather into a list so we can assemble a DataFrame at the end
        row = {"method_key": m_key, "method_name": m_name}
        row.update(stats)   # Insert all of T, N1, pi_hat, LR_uc, p_value, N00, ..., p_value_cc into the row dict
        stats_rows.append(row)

    # === 5. Save the test statistics into a single CSV ====================
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_csv_path = "christoffersen_cc_test_stats.csv"
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"\nSaved backtest statistics (Kupiec + Christoffersen) to: {stats_csv_path}")
        print(stats_df)
        
    # === 6. ES tail-risk analysis & save to CSV ===========================
    # Using the results computed above (hist / param / mc),
    # compare ES on the true loss series over the worst (1-alpha)% tail
    if results:
        es_stats_df = es_tail_summary(
            results_dict=results,
            alpha=alpha,       # ES confidence level (same as VaR)
            tail_alpha=alpha,  # Quantile used to define the tail (change to 0.99 etc. if needed)
        )
        es_stats_csv_path = "es_tail_analysis_stats.csv"
        es_stats_df.to_csv(es_stats_csv_path, index=False)
        print(f"\nSaved ES tail-risk analysis statistics to: {es_stats_csv_path}")
        print(es_stats_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
