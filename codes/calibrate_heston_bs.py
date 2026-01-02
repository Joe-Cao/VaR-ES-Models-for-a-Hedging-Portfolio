# -*- coding: utf-8 -*-
"""
@author: Jianpeng Cao
"""

# calibrate_heston_bs.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from math import log, sqrt, exp, pi
from scipy.optimize import least_squares, brentq
from scipy.integrate import quad

# ===========
# 0) Black–Scholes utilities
# ===========
from math import erf
SQRT2 = sqrt(2.0)
def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT2))

def bs_price(S, K, T, r, q, vol, cp: str):
    """Spot-world Black–Scholes price for a European option (cp: 'C' or 'P')."""
    if T <= 0 or vol <= 0:
        # Degenerate case: at expiry or zero volatility
        intrins = max(0.0, (S - K)) if cp == 'C' else max(0.0, (K - S))
        return intrins * exp(-r*T)
    d1 = (log(S/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    disc_r = exp(-r*T); disc_q = exp(-q*T)
    if cp == 'C':
        return disc_q*S*_std_norm_cdf(d1) - disc_r*K*_std_norm_cdf(d2)
    else:
        return disc_r*K*_std_norm_cdf(-d2) - disc_q*S*_std_norm_cdf(-d1)

def bs_vega(S, K, T, r, q, vol):
    """BS Vega (first derivative w.r.t. volatility, in price units, not annualized %)."""
    if T <= 0 or vol <= 0:
        return 0.0
    d1 = (log(S/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrt(T))
    disc_q = exp(-q*T)
    # Standard normal pdf
    phi = np.exp(-0.5*d1*d1)/sqrt(2*pi)
    return disc_q * S * phi * sqrt(T)

def implied_vol_from_price(S, K, T, r, q, price, cp: str, a=1e-9, b=5.0) -> Optional[float]:
    """Solve for implied vol via Brent; return None if price violates no-arbitrage bounds."""
    # No-arbitrage bounds (discounted-probability bounds)
    lower = max(0.0, (S*exp(-q*T) - K*exp(-r*T)) if cp=='C' else (K*exp(-r*T) - S*exp(-q*T)))
    upper = S*exp(-q*T) if cp=='C' else K*exp(-r*T)
    if not (lower - 1e-12 <= price <= upper + 1e-12):
        return None

    def f(sig):
        return bs_price(S,K,T,r,q,sig,cp) - price

    try:
        # Try to bracket the root at small/large sigma first
        fa, fb = f(a), f(b)
        if fa*fb > 0:
            # Automatically expand the upper bound if needed
            ub = b
            for _ in range(5):
                ub *= 2.0
                fb = f(ub)
                if fa*fb <= 0:
                    b = ub
                    break
        root = brentq(f, a, b, maxiter=100, xtol=1e-10)
        return float(root)
    except Exception:
        return None


def estimate_forward_and_r_by_regression(day_raw: pd.DataFrame, expiry: pd.Timestamp,
                                         use_weights=True):
    """
    For a single maturity T, run a linear regression (C - P) = a + b*K.
    Return F, r, (r - q), and the number of strikes used.
    """
    g = day_raw[day_raw['expiry'] == expiry].copy()
    if g.empty: return None, None, None, 0

    # mid construction + filters
    #if 'mid' not in g.columns:
    #    bid = pd.to_numeric(g.get('bid'), errors='coerce')
    #    ask = pd.to_numeric(g.get('ask'), errors='coerce')
    #    g['mid'] = 0.5*(bid+ask)
    #g['cp'] = g['type'].astype(str).str.upper().str.strip().str[0]
    #g['strike'] = pd.to_numeric(g['strike'], errors='coerce')
    g = g[(g['mid']>0) & np.isfinite(g['strike'])]

    cc = g[g['cp']=='C'][['strike','mid']].rename(columns={'mid':'c_mid'})
    pp = g[g['cp']=='P'][['strike','mid']].rename(columns={'mid':'p_mid'})
    j = pd.merge(cc, pp, on='strike', how='inner')
    if j.empty: return None, None, None, 0

    y = (j['c_mid'] - j['p_mid']).to_numpy(float)    # 目标：C-P
    X = np.c_[np.ones(len(j)), j['strike'].to_numpy(float)]  # [1, K]
    # Optional weights: e.g., inverse-squared relative spread if bid/ask present; use equal weights here
    if use_weights and {'last_bid_price','last_ask_price'}.issubset(g.columns):
        # Simple placeholder (customize for your data)
        w = np.ones_like(y)
    else:
        w = np.ones_like(y)

    # Weighted least squares solution
    WX = X * w[:,None]
    WY = y * w
    beta, *_ = np.linalg.lstsq(WX, WY, rcond=None)   # beta = [a, b]
    a, b = beta
    D_r = -b                       # e^{-rT} = -b
    if not np.isfinite(D_r) or D_r<=0: return None, None, None, len(j)

    # Spot and T
    S = float(np.nanmedian(day_raw['underlying_close']))
    T = (expiry - day_raw['quote_date'].iloc[0]).days / 365.0
    if not (np.isfinite(S) and S>0 and T>0): return None, None, None, len(j)

    F = a / D_r                    # F = (S e^{-qT}) / e^{-rT}
    r = -np.log(D_r)/T
    rq = np.log(F/S)/T             # r - q
    return r, r-rq

# ===========
# 2) Heston semi-analytical pricing (original formula)
# ===========
@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

def _heston_char(type_flag, u, T, params: HestonParams, r, q, S):
    kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0
    # Heston (1993), using the "Little Heston Trap" parameterization (for numerical stability)
    x = log(S)
    a = kappa*theta
    if type_flag==1:
        u_1 = 0.5
        b = kappa - rho*sigma
    else: 
        u_1 = -0.5
        b = kappa
    # d(u)
    d = np.sqrt((rho*sigma*1j*u - b)**2 - (sigma**2) * (1j*u*u_1*2 - u**2))
    # g(u)
    g = (b - rho*sigma*1j*u - d) / (b - rho*sigma*1j*u + d)
    # avoid log(0)
    one_minus_g_exp = 1 - g*np.exp(-d*T)
    one_minus_g     = 1 - g
    C = (r - q)*1j*u*T + a/(sigma**2) * ((b - rho*sigma*1j*u - d)*T - 2.0*np.log(one_minus_g_exp/one_minus_g))
    D = ((b - rho*sigma*1j*u - d) / (sigma**2)) * (1 - np.exp(-d*T)) / one_minus_g_exp
    # Characteristic function
    return np.exp(C + D*v0 + 1j*u*x)

def _heston_Pj(type_flag, S, K, T, r, q, params: HestonParams):
    """P1/P2 in the original Heston formulation (computed via integration), cf. Heston (1993)."""
    # Integrand
    def integrand(u):
        u = float(u)
        cf = _heston_char(type_flag, u, T, params, r, q, S)  # shift for P1
        numerator = np.exp(-1j*u*np.log(K)) * cf
        return np.real(numerator / (1j*u))
    # Numerical integration over (0, ∞). In practice quad is sufficient (Gauss–Laguerre is a faster alternative).
    val, _ = quad(integrand, 0.0, np.inf, limit=200, epsabs=1e-8, epsrel=1e-6)
    return 0.5 + (1.0/pi)*val

def heston_price_call(S, K, T, r, q, params: HestonParams):
    P1 = _heston_Pj(type_flag=1, S=S, K=K, T=T, r=r, q=q, params=params)
    P2 = _heston_Pj(type_flag=2, S=S, K=K, T=T, r=r, q=q, params=params)
    return S*exp(-q*T)*P1 - K*exp(-r*T)*P2

def heston_price(S, K, T, r, q, params: HestonParams, cp: str):
    c = heston_price_call(S,K,T,r,q,params)
    if cp.upper() == 'C':
        return c
    else:
        # Put via parity
        return c - S*exp(-q*T) + K*exp(-r*T)

# ===========
# 3) Objective (price RMSE, optionally weighted)
# ===========
def residuals_heston(params_vec, rows, use_weights=True, price_scale=1.0):
    kappa, theta, sigma, rho, v0 = params_vec
    p = HestonParams(kappa, theta, sigma, rho, v0)
    res = []
    for r_ in rows:
        S, K, T, r_rate, q_rate, mid, cp = r_['S'], r_['K'], r_['T'], r_['r'], r_['q'], r_['mid'], r_['cp']
        w = r_.get('w', 1.0)
        model = heston_price(S, K, T, r_rate, q_rate, p, cp)
        diff = (model - mid)/price_scale
        res.append( (diff * (sqrt(w) if use_weights else 1.0)) )
    return np.array(res, dtype=float)

# ===========
# 4) Main daily calibration function
# ===========
def calibrate_heston_one_day(day_sel: pd.DataFrame,
                             full_day_raw: Optional[pd.DataFrame]=None,
                             r_const: float=0.0,
                             q_const: float=0.0,
                             use_forward_to_infer_carry: bool=True,
                             weight_col: Optional[str]='liq_weight',
                             price_scale: float=1.0,
                             init_guess: Tuple[float,float,float,float,float]=(2.0, 0.04, 0.5, -0.5, 0.04),
                             bounds: Tuple[Tuple,Tuple]=((1e-4, 1e-6, 1e-4, -0.999, 1e-6),
                                                         (20.0,  1.0,  5.0,   0.999, 2.0)) ) -> Dict:
    """
    day_sel: Representative samples from the “point picker” for the day (columns: spot, strike, t_selected, mid, cp).
    full_day_raw: (optional) The day’s full option chain (with cp/strike/mid), used to estimate forward via put–call parity.
    r_const/q_const: If no curves are available, default r=0, q=0. If use_forward_to_infer_carry=True, infer (r - q) from F/S.
    price_scale: For numerical stability you can rescale prices (e.g., divide by S).
    """
    # Assemble rows
    rows = []
    for _, row in day_sel.iterrows():
        S = float(row['spot']);  K = float(row['strike']);  T = float(row['t_selected'])
        mid = float(row['mid']); cp = str(row['cp']).upper()
        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and np.isfinite(mid)):
            continue
        # Rates/dividend: prefer (r - q) inferred from forward if available
        r_rate, q_rate = r_const, q_const
        if use_forward_to_infer_carry and full_day_raw is not None:
            r_rate, q_rate = estimate_forward_and_r_by_regression(full_day_raw, row['expiry'])

        w = 1.0
        if weight_col and weight_col in day_sel.columns and np.isfinite(row.get(weight_col, np.nan)):
            w = float(row[weight_col])

        rows.append({'S':S,'K':K,'T':T,'r':r_rate,'q':q_rate,'mid':mid,'cp':cp,'w':w})

    if len(rows) < 8:
        return {'success': False, 'message':'too few points', 'n':len(rows)}

    # Objective
    def fun(x):
        return residuals_heston(x, rows, use_weights=True, price_scale=price_scale)

    # Optional Feller soft penalty: max(0, sigma^2 - 2 kappa theta)
    def fun_with_penalty(x):
        res = fun(x)
        kappa, theta, sigma = x[0], x[1], x[2]
        feller_violation = max(0.0, sigma*sigma - 2.0*kappa*theta)
        #if feller_violation > 0:
        #    res = np.append(res, 1000.0 * feller_violation)  # 惩罚系数可调
        return res

    # Nonlinear least squares
    ls = least_squares(fun_with_penalty, x0=np.array(init_guess, float),
                       bounds=bounds, max_nfev=200, verbose=0, xtol=1e-10, ftol=1e-10)

    pars = HestonParams(*ls.x)
    # Fit error metrics
    resid = residuals_heston(ls.x, rows, use_weights=False, price_scale=1.0)
    rmse = float(np.sqrt(np.mean(np.square(resid))))
    mae  = float(np.mean(np.abs(resid)))
    return {
        'success': ls.success,
        'message': ls.message,
        'params': pars,
        'rmse_price': rmse,
        'mae_price': mae,
        'n': len(rows),
        'cost': float(ls.cost)
    }

# ===========
# 5) Driver: day-by-day calibration
# ===========
def run_calibration(selected_csv: str,
                    full_raw_csv: Optional[str]=None,
                    start: Optional[str]=None,
                    end: Optional[str]=None,
                    r_const: float=0.0,
                    q_const: float=0.0,
                    use_forward_to_infer_carry: bool=True) -> pd.DataFrame:

    sel = pd.read_csv(selected_csv, parse_dates=['quote_date','expiry'])
    if start: sel = sel[sel['quote_date'] >= pd.to_datetime(start)]
    if end:   sel = sel[sel['quote_date'] <= pd.to_datetime(end)]

    full = None
    if full_raw_csv:
        full = pd.read_csv(full_raw_csv, parse_dates=['quote_date','expiry'])
        # Ensure required columns / cleaning
        full.columns = [c.strip().lower() for c in full.columns]
        # Canonical columns
        if 'last_bid_price' in full.columns: full = full.rename(columns={'last_bid_price':'bid'})
        if 'last_ask_price' in full.columns: full = full.rename(columns={'last_ask_price':'ask'})
        full['mid'] = np.where(
            np.isfinite(full.get('bid')) & np.isfinite(full.get('ask')) & (full['bid']>0) & (full['ask']>0),
            0.5*(full['bid']+full['ask']),
            pd.to_numeric(full.get('last', np.nan), errors='coerce')
        )
        if 'type' in full.columns:
            full['cp'] = full['type'].astype(str).str.upper().str[0]
        full['strike'] = pd.to_numeric(full['strike'], errors='coerce')

    results = []
    for d, day_sel in sel.groupby('quote_date'):
        day_sel = day_sel.copy()
        # Enforce required columns
        needed = ['spot','strike','t_selected','mid','cp','expiry']
        if not all(c in day_sel.columns for c in needed):
            print(f"[{d.date()}] missing columns in selected file")
            continue

        # full day raw for parity (same date)
        full_day = None
        if full is not None:
            full_day = full[full['quote_date']==d].copy()

        out = calibrate_heston_one_day(
            day_sel,
            full_day_raw=full_day,
            r_const=r_const, q_const=q_const,
            use_forward_to_infer_carry=use_forward_to_infer_carry
        )
        res = {
            'quote_date': d,
            'success': out['success'],
            'message': out['message'],
            'kappa': getattr(out['params'],'kappa', np.nan),
            'theta': getattr(out['params'],'theta', np.nan),
            'sigma': getattr(out['params'],'sigma', np.nan),
            'rho':   getattr(out['params'],'rho',   np.nan),
            'v0':    getattr(out['params'],'v0',    np.nan),
            'rmse_price': out.get('rmse_price', np.nan),
            'mae_price' : out.get('mae_price',  np.nan),
            'n_points'  : out.get('n', 0),
        }
        results.append(res)

    res_df = pd.DataFrame(results).sort_values('quote_date').reset_index(drop=True)
    return res_df

# ===========
# 6) Main script: example usage
# ===========
if __name__ == "__main__":
    # Replace these paths with your own files
    SELECTED_CSV = "selected_bs_points.csv"         # Output from your “point picker (BS version)”
    FULL_RAW_CSV = "your_full_chain.csv"            # Full raw chain (optional, for forward estimation via parity)
    USE_FULL = False                                # Set False if you do not have a full chain

    res = run_calibration(
        selected_csv=SELECTED_CSV,
        full_raw_csv=FULL_RAW_CSV if USE_FULL else None,
        start=None, end=None,
        r_const=0.0,    # If you have an OIS curve, set r to the day’s OIS rate
        q_const=0.0,    # If you have dividend/borrow curves, set q accordingly
        use_forward_to_infer_carry=USE_FULL # When full chain is provided, estimate (r - q) from parity
    )
    print(res.head())
    res.to_csv("heston_params_daily.csv", index=False)
    print("Saved -> heston_params_daily.csv")
