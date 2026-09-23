# Long-Term Trading Bot

A bot that looks for stocks worth holding long term by running several models side by side: news sentiment from an LLM, a GBM price simulation, a CNN-LSTM forecaster (CLAM), and an LPPL check for market bubbles. The model research lives in [Quant_Model_Research](https://github.com/nahniee/Quant_Model_Research).

I later reviewed the GBM and CLAM models in [Signal_Validation](https://github.com/nahniee/Signal_Validation). That review led to `gbm_weekly.py`, a weekly version of the GBM score.

## Models

| Model | What it's for | How it works |
|-------|---------------|--------------|
| LLM sentiment | News sentiment | Labels each ticker's news articles as positive, negative or neutral with an LLM |
| GBM simulation | Price paths | Simulates a price path with geometric Brownian motion |
| CLAM | Growth forecast | CNN + LSTM + attention model that forecasts the next 65 days |
| Bayesian LPPL | Crash timing | Bayesian fit of the log-periodic power law to estimate when a bubble might end |

## Pipeline

1. News sentiment. The bot pulls recent articles from Yahoo Finance and asks an LLM (OpenAI) to label each one positive, negative or neutral. A ticker's score is (positive - negative) / total articles.

2. GBM path simulation. For each ticker it estimates drift and volatility from about two years of log returns, simulates one 63-day path, and scores the stock by how far the path's average sits above today's price.

3. CLAM forecast. The model takes 252 days of daily price changes, forecasts the next 65 days and turns that into an expected growth figure.

4. Rankings. The bot prints a ranking for each of the three models. The step that combines them is commented out in `long_term_trading.py` for now; the planned weighting is:
   ```
   Weighted Score = (News Rank x 0.45) + (CLAM Rank x 0.35) + (GBM Rank x 0.20)
   ```

5. LPPL. The bot runs a Bayesian MCMC fit on the Nasdaq (`^IXIC`) and the S&P 500 (`^GSPC`) and plots the most likely crash date with a 94% highest density interval.
