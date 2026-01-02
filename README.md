# VaR/ES Models for a Delta-Hedged Portfolio Using the Heston Pricing Mechanism

This project delivers an end-to-end pipeline for Value-at-Risk (VaR) and Expected Shortfall (ES) modeling of a delta-hedged strategy using the Heston option-pricing model, including data preparation, model implementation, backtesting, and tail-risk analysis. It is a continuation of *Calibrating the Heston Model for Option Pricing*.

## Data Preprocessing

We merge the Heston-parameter table with the PnL table for downstream tasks. See data_utils.py for details.

## VaR Model Implementation

We implement Historical VaR, Variance–Covariance (parametric) VaR, and Monte Carlo (bootstrap) VaR to estimate VaR and ES for each trade date under a given confidence level. See var_models.py.

## VaR Backtesting

For each day, we check whether the realized loss exceeds the model-predicted VaR, compute the violation frequency, and run Kupiec and Christoffersen tests for every VaR model. See backtesting.py, stats_tests.py, and run_var_backtest.py.

## ES Tail-Risk Analysis

At a chosen confidence level, we compare each model’s theoretical ES over the worst tail with the realized ES to assess which model is more tail-sensitive or more conservative. See es_tail_analysis.py.
