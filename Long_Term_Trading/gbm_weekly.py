"""GBM weekly ranking, v2 (redeveloped after the Signal_Validation review).

Changes vs get_gbm_path_simulation (quarterly):
  * horizon 63 -> 5 trading days (weekly rebalance)
  * the single Monte-Carlo path is replaced by the closed-form GBM quantities,
    which removes the sigma*sqrt(T) noise that was destroying the ranking
  * two scores are available:
        expected  : E[S_h / S_0] - 1           = exp(mu_d * h) - 1
        prob_up   : P(S_h > S_0)               = Phi((mu_d - sigma_d^2 / 2) * sqrt(h) / sigma_d)
    prob_up is a risk-adjusted rank (drift per unit of volatility).
Inputs are close prices; mu_d, sigma_d are daily log-return mean and std over `lookback` days.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

HORIZON_DAYS = 5


def gbm_weekly_score(close: pd.Series, lookback: int = 126, mode: str = "prob_up") -> float:
    lr = np.log(close).diff().dropna().iloc[-lookback:]
    if len(lr) < int(lookback * 0.9):
        return np.nan
    mu, sigma = lr.mean(), lr.std()
    if mode == "expected":
        return float(np.exp(mu * HORIZON_DAYS) - 1.0)
    if mode == "prob_up":
        return float(norm.cdf((mu - 0.5 * sigma ** 2) * np.sqrt(HORIZON_DAYS) / sigma)) if sigma > 0 else np.nan
    raise ValueError(mode)


if __name__ == "__main__":
    import yfinance as yf
    px = yf.download("AMD", period="1y", auto_adjust=True, progress=False)["Close"].squeeze()
    for m in ("expected", "prob_up"):
        print(m, round(gbm_weekly_score(px, mode=m), 4))
