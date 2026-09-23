"""Weekly GBM ranking, rewritten after the Signal_Validation review.

Compared with get_gbm_path_simulation (quarterly), the horizon drops from 63 trading
days to 5 to match a weekly rebalance, and the score comes from closed-form GBM results
instead of one simulated path. The single path added noise of order sigma*sqrt(T),
which was scrambling the ranking.

Two scores are available:
    expected : E[S_h / S_0] - 1 = exp(mu_d * h) - 1
    prob_up  : P(S_h > S_0)     = Phi((mu_d - sigma_d^2 / 2) * sqrt(h) / sigma_d)
prob_up ranks by drift per unit of volatility.

Inputs are close prices; mu_d and sigma_d are the mean and standard deviation of daily
log returns over `lookback` days.
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
