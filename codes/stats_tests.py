# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 04:44:13 2025

@author: caoxi
"""

# stats_tests.py
from __future__ import annotations
import math
import numpy as np
from scipy.stats import chi2


def kupiec_test(violations, alpha: float):
    """
    Kupiec Unconditional Coverage Test.

    Parameters
    ----------
    violations : array-like of 0/1
        1 indicates that on that day L_t > VaR_t.
    alpha : float
        VaR confidence level (e.g., 0.95).

    Returns
    -------
    dict
        Keys: T, N1, pi_hat, LR_uc, p_value
    """
    v = np.asarray(violations, dtype=int)
    T = v.size
    N1 = v.sum()
    pi_hat = N1 / T if T > 0 else np.nan
    p = 1.0 - alpha  # theoretical tail probability

    if T == 0:
        return {"T": T, "N1": N1, "pi_hat": pi_hat,
                "LR_uc": np.nan, "p_value": np.nan}

    # Add a small epsilon to avoid log(0)
    eps = 1e-10
    p0 = min(max(p, eps), 1 - eps)
    ph = min(max(pi_hat, eps), 1 - eps)

    L0 = (T - N1) * math.log(1 - p0) + N1 * math.log(p0)
    L1 = (T - N1) * math.log(1 - ph) + N1 * math.log(ph)
    LR_uc = -2.0 * (L0 - L1)
    p_value = chi2.sf(LR_uc, df=1)

    return {
        "T": T,
        "N1": int(N1),
        "pi_hat": pi_hat,
        "LR_uc": LR_uc,
        "p_value": p_value,
    }


def christoffersen_independence_test(violations):
    """
    Christoffersen independence test (checks for violation clustering).

    Returns
    -------
    dict
        Keys: N00, N01, N10, N11, LR_ind, p_value
    """
    v = np.asarray(violations, dtype=int)
    if v.size < 2:
        return {
            "N00": np.nan,
            "N01": np.nan,
            "N10": np.nan,
            "N11": np.nan,
            "LR_ind": np.nan,
            "p_value": np.nan,
        }

    N00 = N01 = N10 = N11 = 0
    for i in range(1, len(v)):
        prev, cur = v[i - 1], v[i]
        if prev == 0 and cur == 0:
            N00 += 1
        elif prev == 0 and cur == 1:
            N01 += 1
        elif prev == 1 and cur == 0:
            N10 += 1
        elif prev == 1 and cur == 1:
            N11 += 1

    denom0 = N00 + N01
    denom1 = N10 + N11
    if denom0 == 0 or denom1 == 0:
        return {
            "N00": N00,
            "N01": N01,
            "N10": N10,
            "N11": N11,
            "LR_ind": np.nan,
            "p_value": np.nan,
        }

    pi0_hat = N01 / denom0
    pi1_hat = N11 / denom1
    pi_hat = (N01 + N11) / (denom0 + denom1)

    def safe_log(x):
        return math.log(x) if x > 0 else -1e9

    L0 = (
        N00 * safe_log(1 - pi_hat)
        + N01 * safe_log(pi_hat)
        + N10 * safe_log(1 - pi_hat)
        + N11 * safe_log(pi_hat)
    )
    L1 = (
        N00 * safe_log(1 - pi0_hat)
        + N01 * safe_log(pi0_hat)
        + N10 * safe_log(1 - pi1_hat)
        + N11 * safe_log(pi1_hat)
    )

    LR_ind = -2.0 * (L0 - L1)
    p_value = chi2.sf(LR_ind, df=1)

    return {
        "N00": int(N00),
        "N01": int(N01),
        "N10": int(N10),
        "N11": int(N11),
        "LR_ind": LR_ind,
        "p_value": p_value,
    }


def christoffersen_cc_test(violations, alpha: float):
    """
    Conditional coverage test: LR_cc = LR_uc + LR_ind, ~ chi2(df=2).
    This combines both coverage and independence dimensions.
    """
    k = kupiec_test(violations, alpha)
    ind = christoffersen_independence_test(violations)

    if math.isnan(ind["LR_ind"]) or math.isnan(k["LR_uc"]):
        LR_cc = np.nan
        p_cc = np.nan
    else:
        LR_cc = k["LR_uc"] + ind["LR_ind"]
        p_cc = chi2.sf(LR_cc, df=2)

    out = {}
    out.update(k)
    out.update(
        {
            "N00": ind["N00"],
            "N01": ind["N01"],
            "N10": ind["N10"],
            "N11": ind["N11"],
            "LR_ind": ind["LR_ind"],
            "p_value_ind": ind["p_value"],
            "LR_cc": LR_cc,
            "p_value_cc": p_cc,
        }
    )
    return out
