
# -*- coding: utf-8 -*-
"""
@Author: Jianpeng Cao

Delta-hedging backtest (stepwise PnL) tailored to your two CSV layouts:
- full_raw_csv (1st file): has columns like
  ['quote_date','underlying_symbol','root','expiry','strike','type',
   'open_interest','total_volume','high','low','open','last',
   'last_bid_price','last_ask_price','underlying_close', ...]
- selected_csv (2nd file): has columns like
  ['quote_date','expiry','strike','cp','mid','bid','ask','spread_abs','spread_pct',
   'open_interest','total_volume','underlying_symbol','spot','k','k_target','k_distance',
   'prefer_cp','liq_weight']

Notes:
- We DO NOT require 'last'/'bid'/'ask' in selected_csv; if missing, we will use its 'mid' directly.
- For full_raw_csv next-day prices we compute mid = (last_bid_price + last_ask_price)/2 if both present;
  otherwise fall back to 'last' if available.
- Time-to-maturity T_i is computed from (expiry - quote_date)/365.0 in selected_csv (no need for t_selected).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from math import log, sqrt, pi
from typing import Optional, Dict, Tuple
from scipy.integrate import quad
from calibrate_heston_bs import *

def heston_delta_fd(S, K, T, r, q, p: HestonParams, cp: str, rel_eps: float = 1e-4):
    h = max(1e-6, abs(S) * rel_eps)
    up = heston_price(S + h, K, T, r, q, p, cp)
    dn = heston_price(S - h, K, T, r, q, p, cp)
    return (up - dn) / (2.0 * h)

# ==============================
# Helpers specific to your CSVs
# ==============================
def normalize_full_raw(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    for c in ['strike','last','last_bid_price','last_ask_price','underlying_close']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    for dc in ['quote_date','expiry']:
        d[dc] = pd.to_datetime(d[dc], errors='coerce')
    # cp from 'type'
    if 'cp' not in d.columns:
        if 'type' in d.columns:
            s = d['type'].astype(str).str.upper().str.strip()
            s = s.replace({'CALL':'C','PUT':'P'})
            d['cp'] = s.str[0]
    # spot
    d['spot'] = pd.to_numeric(d.get('underlying_close', np.nan), errors='coerce')
    # mid from last_bid/ask or last
    bid = pd.to_numeric(d.get('last_bid_price', np.nan), errors='coerce')
    ask = pd.to_numeric(d.get('last_ask_price', np.nan), errors='coerce')
    mid = pd.Series(np.nan, index=d.index, dtype='float64')
    mask = bid.gt(0) & ask.gt(0)
    mid.loc[mask] = 0.5*(bid.loc[mask] + ask.loc[mask])
    if 'last' in d.columns:
        mid = mid.fillna(d['last'])
    d['mid'] = mid
    return d

def normalize_selected(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    for c in ['strike','mid','bid','ask','spot']:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors='coerce')
    for dc in ['quote_date','expiry']:
        d[dc] = pd.to_datetime(d[dc], errors='coerce')
    # ensure cp
    if 'cp' in d.columns:
        d['cp'] = d['cp'].astype(str).str.upper().str.strip().str[0]
    elif 'type' in d.columns:
        s = d['type'].astype(str).str.upper().str.strip().replace({'CALL':'C','PUT':'P'})
        d['cp'] = s.str[0]
    else:
        raise ValueError("selected_csv must have 'cp' or 'type'.")
    # ensure mid
    if 'mid' not in d.columns or d['mid'].isna().all():
        # try bid/ask in selected
        if 'bid' in d.columns and 'ask' in d.columns:
            b = pd.to_numeric(d['bid'], errors='coerce')
            a = pd.to_numeric(d['ask'], errors='coerce')
            m = pd.Series(np.nan, index=d.index, dtype='float64')
            m.loc[b.gt(0)&a.gt(0)] = 0.5*(b.where(b>0)+a.where(a>0))
            d['mid'] = m
    return d

def estimate_forward_by_parity(day_raw: pd.DataFrame, expiry: pd.Timestamp) -> Optional[float]:
    g = day_raw[day_raw['expiry'] == expiry].copy()
    if g.empty: return None
    cc = g[g['cp']=='C'][['strike','mid']].rename(columns={'mid':'c_mid'})
    pp = g[g['cp']=='P'][['strike','mid']].rename(columns={'mid':'p_mid'})
    j = pd.merge(cc, pp, on='strike', how='inner')
    if j.empty: return None
    F_est = j['strike'] + (j['c_mid'] - j['p_mid'])
    return float(F_est.median())

def infer_r_minus_q(S, F, T):
    if S<=0 or F is None or F<=0 or T<=0: return 0.0
    return np.log(F/S)/T

# ==============================
# Core backtest
# ==============================
def run_backtest(selected_csv: str,
                 full_raw_csv: str,
                 calib_csv: str,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 r_const: float = 0.0,
                 q_const: float = 0.0,
                 use_forward_to_infer_carry: bool = True,
                 tc_bps_per_leg: float = 0.0,
                 cash_rate: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sel0 = pd.read_csv(selected_csv)
    raw0 = pd.read_csv(full_raw_csv)

    sel = normalize_selected(sel0)
    raw = normalize_full_raw(raw0)

    if start: sel = sel[sel['quote_date'] >= pd.to_datetime(start)]
    if end:   sel = sel[sel['quote_date'] <= pd.to_datetime(end)]

    # fill spot for selected if missing
    raw_spot = raw[['quote_date','spot']].dropna().drop_duplicates()
    sel = sel.merge(raw_spot, on='quote_date', how='left', suffixes=('', '_raw'))
    if 'spot' not in sel.columns or sel['spot'].isna().all():
        sel['spot'] = sel['spot_raw']
    else:
        sel['spot'] = sel['spot'].fillna(sel['spot_raw'])
    if 'spot_raw' in sel.columns: sel.drop(columns=['spot_raw'], inplace=True)

    # calib
    calib = pd.read_csv(calib_csv, parse_dates=['quote_date'])
    for c in ['kappa','theta','sigma','rho','v0']:
        if c not in calib.columns:
            raise ValueError(f"Missing column {c} in calib_csv")
    calib['quote_date'] = pd.to_datetime(calib['quote_date']).dt.normalize()
    calib['params'] = calib[['kappa','theta','sigma','rho','v0']].apply(lambda s: HestonParams(*map(float, s)), axis=1)
    calib_idx: Dict[pd.Timestamp, HestonParams] = dict(zip(calib['quote_date'], calib['params']))

    # spot index
    spot_idx: Dict[pd.Timestamp, float] = dict(zip(raw['quote_date'].dt.normalize(), raw['spot']))

    legs = []
    for d in sorted(sel['quote_date'].dt.normalize().unique()):
        d_next = d + pd.Timedelta(days=1)
        if d not in calib_idx or d_next not in calib_idx:
            continue
        if d not in spot_idx or d_next not in spot_idx:
            continue

        Theta_i, Theta_ip1 = calib_idx[d], calib_idx[d_next]
        S_i, S_ip1 = float(spot_idx[d]), float(spot_idx[d_next])

        day_sel = sel[sel['quote_date'].dt.normalize()==d].copy()
        if day_sel.empty: continue
        raw_i   = raw[raw['quote_date'].dt.normalize()==d].copy()
        raw_ip1 = raw[raw['quote_date'].dt.normalize()==d_next].copy()            

        for _, r_ in day_sel.iterrows():
            cp = str(r_['cp']).upper()[0]
            K  = float(r_['strike'])
            expi = pd.to_datetime(r_['expiry']).normalize()
            T_i   = max(0.0, (expi - d).days / 365.0)      # compute from dates
            T_ip1 = max(0.0, (expi - d_next).days / 365.0) # next-day residual
            if T_i <= 0: 
                continue

            V_i_mkt = float(r_.get('mid', np.nan))
            if not np.isfinite(V_i_mkt):
                # try selected bid/ask
                if np.isfinite(r_.get('bid', np.nan)) and np.isfinite(r_.get('ask', np.nan)) and r_['bid']>0 and r_['ask']>0:
                    V_i_mkt = 0.5*(float(r_['bid'])+float(r_['ask']))
                else:
                    continue  # no price

            # find next-day same contract in raw_ip1 (mid built already)
            m2 = raw_ip1[(raw_ip1['expiry']==expi) & (raw_ip1['strike']==K) & (raw_ip1['cp']==cp)]
            if m2.empty or not np.isfinite(m2.iloc[0]['mid']):
                continue
            V_ip1_mkt = float(m2.iloc[0]['mid'])

            r_rate, q_rate = r_const, q_const
            if use_forward_to_infer_carry and raw_i is not None:
                r_rate, q_rate = estimate_forward_and_r_by_regression(raw_i, expi)

            p_i, p_ip1 = Theta_i, Theta_ip1

            # Stepwise revaluation
            V_model_t_i    = heston_price(S_i,   K, T_i,   r_rate, q_rate, p_i,   cp)
            V_model_t_ip1a = heston_price(S_i,   K, T_ip1, r_rate, q_rate, p_i,   cp)
            TH = V_model_t_ip1a - V_model_t_i

            V_model_t_ip1b = heston_price(S_ip1, K, T_ip1, r_rate, q_rate, p_i,   cp)
            UM = V_model_t_ip1b - V_model_t_ip1a

            V_model_t_ip1c = heston_price(S_ip1, K, T_ip1, r_rate, q_rate, p_ip1, cp)
            SM = V_model_t_ip1c - V_model_t_ip1b

            eps_i   = V_i_mkt   - V_model_t_i
            eps_ip1 = V_ip1_mkt - V_model_t_ip1c
            ME = eps_ip1 - eps_i

            Delta_i = heston_delta_fd(S_i, K, T_i, r_rate, q_rate, p_i, cp)
            dS = S_ip1 - S_i

            TC = 0.0
            if tc_bps_per_leg and tc_bps_per_leg > 0:
                TC = (tc_bps_per_leg * 1e-4) * (abs(V_i_mkt) + abs(Delta_i) * S_i)

            cash_pnl = cash_rate * (1.0/252.0)
            pnl_total = (TH + (UM - Delta_i*dS) + SM + ME) + cash_pnl - TC

            legs.append({
                'date': d, 'date_next': d_next, 'expiry': expi, 'cp': cp, 'strike': K,
                'S_i': S_i, 'S_ip1': S_ip1, 'T_i': T_i, 'T_ip1': T_ip1,
                'V_i_mkt': V_i_mkt, 'V_ip1_mkt': V_ip1_mkt,
                'r': r_rate, 'q': q_rate,
                'kappa_i': p_i.kappa, 'theta_i': p_i.theta, 'sigma_i': p_i.sigma, 'rho_i': p_i.rho, 'v0_i': p_i.v0,
                'kappa_ip1': p_ip1.kappa, 'theta_ip1': p_ip1.theta, 'sigma_ip1': p_ip1.sigma, 'rho_ip1': p_ip1.rho, 'v0_ip1': p_ip1.v0,
                'Delta_i': Delta_i,
                'TH': TH, 'UM_raw': UM, 'UM_delta_hedged': (UM - Delta_i * dS),
                'SM': SM, 'ME': ME, 'TC': TC, 'cash_pnl': cash_pnl,
                'PnL_total': pnl_total
            })

    if not legs:
        return pd.DataFrame(), pd.DataFrame()

    legs_df = pd.DataFrame(legs).sort_values(['date','expiry','cp','strike']).reset_index(drop=True)

    daily_df = legs_df.groupby('date', as_index=False).agg(
        TH=('TH','sum'),
        UM_delta_hedged=('UM_delta_hedged','sum'),
        SM=('SM','sum'),
        ME=('ME','sum'),
        TC=('TC','sum'),
        cash_pnl=('cash_pnl','sum'),
        PnL_total=('PnL_total','sum')
    )
    if len(daily_df) >= 2:
        mu = daily_df['PnL_total'].mean()
        sd = daily_df['PnL_total'].std(ddof=1)
        daily_df.attrs['sharpe_annualized'] = (mu/sd)*np.sqrt(252) if sd>0 else np.nan

    return legs_df, daily_df

# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Delta-hedging backtest (tailored to your full_raw_csv & selected_csv layouts)")
    ap.add_argument("--selected_csv", required=True)
    ap.add_argument("--full_raw_csv", required=True)
    ap.add_argument("--calib_csv",    required=True, help="Daily calibrated Heston params CSV (quote_date,kappa,theta,sigma,rho,v0)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--infer_forward", action="store_true")
    ap.add_argument("--tc_bps", type=float, default=0.0)
    ap.add_argument("--cash_rate", type=float, default=0.0)
    ap.add_argument("--out_prefix", default="backtest")
    args = ap.parse_args()

    legs_df, daily_df = run_backtest(
        selected_csv=args.selected_csv,
        full_raw_csv=args.full_raw_csv,
        calib_csv=args.calib_csv,
        start=args.start, end=args.end,
        r_const=args.r, q_const=args.q,
        use_forward_to_infer_carry=args.infer_forward,
        tc_bps_per_leg=args.tc_bps,
        cash_rate=args.cash_rate
    )

    legs_path  = f"{args.out_prefix}_legs.csv"
    daily_path = f"{args.out_prefix}_daily.csv"
    legs_df.to_csv(legs_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    print(f"Saved legs -> {legs_path}")
    print(f"Saved daily -> {daily_path}")
    if hasattr(daily_df, 'attrs') and 'sharpe_annualized' in daily_df.attrs:
        print(f"Annualized Sharpe: {daily_df.attrs['sharpe_annualized']:.3f}")
