import os
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def bayesian_adjustment(positive, negative, neutral, prior_mean=0, weight=20):
    """
    Apply Bayesian Adjustment to Sentiment Scores.
    - prior_mean: Expected mean sentiment (default 0)
    - weight: Prior weight to balance extreme scores
    """
    total_votes = positive + negative + neutral
    adjusted_sentiment = (positive * 2 - negative * 1 + (prior_mean * weight)) / (total_votes + weight)
    return adjusted_sentiment

def get_gbm_drift_calculation(ticker_name, period_year=1):
    try:
        # 1. Download past 1-year data
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * period_year)
        ticker = yf.Ticker(ticker_name)
        stock = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
        if stock is None or stock.empty:
            print(f"No historical data found for {ticker_name}.")
            return 0
        
        # 2. Calculate log returns
        stock['LogReturn'] = np.log(stock['Close'] / stock['Close'].shift(1))
        stock = stock.dropna()

        # 3. Calculate daily mu and sigma
        mu = stock['LogReturn'].mean()
        sigma = stock['LogReturn'].std()
        
        # 4. Calculate average daily drift
        gbm_drift = (mu - (0.5 * sigma**2))
        return gbm_drift
    except Exception as e:
        print(f"Error calculating GBM drift for {ticker_name}: {e}")
        return 0
    
def get_gbm_path_simulation(ticker_name, mode ="quarterly"):
    """
    Simulate stock price using Geometric Brownian Motion (GBM).
    """
    if mode == 'quarterly':
        params = {
            'period': 365*2,
            'interval': '1d',
            'scaling_factor': 252,  # Trading days per year
            'T': 0.25,              # 0.25 years (1 quarter)
            'N': 63,                # 63 trading days
            'xlabel': 'Trading Day'
        }
    elif mode == 'hourly':
        params = {
            'period': 60,
            'interval': '30m',
            'scaling_factor': 14,   # 14 trading hours per day
            'T': 1,                 # 1 day
            'N': 14,                 # 14 trading hours
            'xlabel': 'Trading Hour of the Day'
        }
    else:
        print(f"Unsupported mode: {mode}. Supported modes are 'quarterly' and 'hourly'.")
        return 0
    
    try:
        # 1. Fetch historical data
        end_date = datetime.today()
        start_date = end_date - timedelta(days=params['period'])
        ticker = yf.Ticker(ticker_name)
        stock = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=params['interval'])
        if stock is None or stock.empty:
            print(f"No historical data found for {ticker_name}.")
            return 0
        
        # 2. Calculate log returns
        stock['LogReturn'] = np.log(stock['Close'] / stock['Close'].shift(1))
        stock = stock.dropna()
        
        # 3. Calculate annualized drift (mu) and volatility (sigma)
        mu = stock['LogReturn'].mean() * params['scaling_factor']
        sigma = stock['LogReturn'].std() * np.sqrt(params['scaling_factor'])
        S0 = stock['Close'].iloc[-1].item()
        dt = params['T'] / params['N']

        brownian_increments = np.random.normal(0, np.sqrt(dt), params['N'])
        brownian_path = np.concatenate([np.array([0]), np.cumsum(brownian_increments)])
        time_grid = np.linspace(0, params['T'], params['N'] + 1)
        gbm_path = S0 * np.exp((mu - 0.5 * sigma**2) * time_grid + sigma * brownian_path)
        
        path_avg = np.mean(gbm_path)
        normalized_score = (path_avg - S0) / S0
        return normalized_score
    
    except Exception as e:
        print(f"Error simulating GBM path for {ticker_name}: {e}")
        return 0

if __name__ == "__main__":
    """
    python -m src.ticker_data_processor.stats_model_process
    """
    # Example usage
    ticker_name = "AMD"
    
    # Get GBM drift
    # gbm_drift = get_gbm_drift_calculation(env_config, ticker_name)
    # print(f"GBM Drift for {ticker_name}: {gbm_drift}")

    gbm = get_gbm_path_simulation(ticker_name, mode="quarterly")
    print(f"GBM Path for {ticker_name}: {gbm}")

    gbm = get_gbm_path_simulation(ticker_name, mode="hourly")
    print(f"GBM Path for {ticker_name} (Hourly): {gbm}")