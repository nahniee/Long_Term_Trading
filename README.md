# Long-Term Trading Bot with Multi-Model Ranking System

This project is a trading bot designed to **automatically identify promising stock tickers for long-term investment** by integrating various quantitative and machine learning models.  
It follows a 4-step analytical pipeline to **forecast market direction, individual ticker growth potential, and even potential market crash timing**.

---

## Model Overview (from `Quant Model Research`)

| Model                 | Purpose                        | Description |
|-----------------------|--------------------------------|-------------|
| `LLM Sentiment Model` | News-based sentiment analysis  | Scores each ticker's news as positive, negative, or neutral using large language models |
| `GBM Simulation`      | Price path forecasting         | Simulates probabilistic price paths using Geometric Brownian Motion |
| `CLAM Model`          | Precise growth prediction      | Uses CNN + LSTM + Attention to forecast future closing prices |
| `Bayesian LPPL`       | Bubble crash timing detection  | Applies Bayesian inference to estimate the critical time of market bubbles or crashes |

---

## Analysis Pipeline

1. **News Sentiment Analysis**  
   - Collects news articles from Yahoo Finance and processes them using an LLM (OpenAI)
   - Each article is classified as Positive / Negative / Neutral
   - Sentiment Score = (Positive - Negative) / Total Number of Articles

2. **GBM Path Simulation**  
   - Simulates Brownian price paths using each ticker’s log returns, drift, and volatility
   - Estimates the likelihood of positive return, which becomes a ranking score

3. **CLAM Inference (CNN + LSTM + Attention)**  
   - Uses 252 days of historical log returns to predict the next 65 days of closing prices
   - Computes expected growth percentage based on predicted future prices

4. **Final Score Aggregation (Weighted Ranking)**  
   - Aggregates model scores using a weighted average to determine final rankings:
     ```
     Weighted Score = (News Rank × 0.45) + (CLAM Rank × 0.35) + (GBM Rank × 0.20)
     ```

5. **LPPL (Log-Periodic Power Law)**  
   - Performs Bayesian MCMC simulations on `^IXIC` (NASDAQ) and `^GSPC` (S&P500)
   - Visualizes the most probable crash date (mode) and the 94% Highest Density Interval (HDI)