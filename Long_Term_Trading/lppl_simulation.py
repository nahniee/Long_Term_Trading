import pymc as pm
import numpy as np
import pandas as pd
import yfinance as yf
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde

def time_price_data(ticker_symbol, year_period = 2):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * year_period)
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']].dropna()
    time_data = data['Date'].apply(pd.Timestamp.toordinal).values

    # Take natural log of prices 
    log_price_data = np.log(data['Close'].values)

    return time_data, log_price_data

def lppl_mcmc(time_data, log_price_data):
    # PyMC Modeling (Reparameterized LPPL)
    with pm.Model() as lppl_model:
        t_end = time_data[-1]   # Last time in the dataset
        tc = pm.Uniform('tc', lower=t_end, upper=t_end + 100, initval=t_end + 50)   # Critical time
        m = pm.Uniform('m', lower=0.1, upper=1.0)   # Exponent for power-law
        w = pm.Uniform('w', lower=4.0, upper=13.0)  # Log-periodic frequency
        A = pm.Normal('A', mu=log_price_data.mean(), sigma=log_price_data.std())    # Baseline level
        B = pm.Normal('B', mu=0, sigma=1)       # Amplitude of power-law component
        c1 = pm.Normal('c1', mu=0, sigma=1)     # Cosine term coefficient
        c2 = pm.Normal('c2', mu=0, sigma=1)     # Sine term coefficient
        sigma = pm.HalfNormal('sigma', sigma=0.1)   # Observation noise

        # Define Expected Value
        dt = tc - time_data      # Time to critical point
        log_dt = pm.math.log(dt)
        oscillation = c1 * pm.math.cos(w * log_dt) + c2 * pm.math.sin(w * log_dt)       # Log-periodic oscillation
        expected_log_price = A + B * (dt**m) + (dt**m) * oscillation        # Full LPPL expression

        # Likelihood: observed log prices follow Normal distribution around model prediction
        pm.Normal('Y_obs', mu=expected_log_price, sigma=sigma, observed=log_price_data)

        trace = pm.sample(
            draws=4000,     # Posterior samples to draw
            tune=2000, 
            target_accept=0.99,     # High target_accept for better stability
            control={'max_treedepth': 15},      # Increase max tree depth for complex posterior geometries
            chains=4,       # Run 4 independent MCMC chains to assess convergence (e.g., via R-hat diagnostics)
            cores=4,        # Use 4 CPU cores to run the chains in parallel for faster sampling
            progressbar=True
        )

    return trace

def summarize_posterior(trace):
    # Summarize posterior samples of tc
    summary = az.summary(trace, var_names=['tc'])
    tc_samples = trace.posterior['tc'].values.flatten()

    # Kernel Density Estimation on tc
    kde = gaussian_kde(tc_samples)
    x_grid = np.linspace(tc_samples.min(), tc_samples.max(), 1000)

    # Mode (most probable tc)
    mode_ordinal = x_grid[np.argmax(kde.evaluate(x_grid))]
    mode_date = datetime.fromordinal(int(mode_ordinal))

    # Extract 94% HDI (highest density interval) bounds for tc
    hdi_3_date = datetime.fromordinal(int(summary.loc['tc', 'hdi_3%']))
    hdi_97_date = datetime.fromordinal(int(summary.loc['tc', 'hdi_97%']))

    return tc_samples, mode_date, hdi_3_date, hdi_97_date


def plot_posterior_tc(tc_samples, mode_date, hdi_3_date, hdi_97_date, ticker_symbol):
    # Plotting Posterior Distribution
    tc_dates = [datetime.fromordinal(int(val)) for val in tc_samples]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram and KDE using seaborn
    sns.histplot(x=tc_dates, ax=ax, bins=50, kde=True, stat="density")

    # Draw HDI interval as a thick black line
    y_pos = ax.get_ylim()[1] * 0.05
    ax.plot([hdi_3_date, hdi_97_date], [y_pos, y_pos], lw=4, color='black', label='94% HDI')
    ax.text(mode_date, y_pos, '94% HDI', ha='center', va='bottom', fontsize=12)
    
    # Add a vertical line at the mode of the distribution
    ax.axvline(mode_date, color='red', linestyle='--', label=f'Mode: {mode_date.strftime("%Y-%m-%d")}')

    # Set plot labels and title
    ax.set_title(f'Posterior Distribution of Critical Time ({ticker_symbol})', fontsize=16)
    ax.set_xlabel('Predicted Critical Time (tc)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Rotate date labels for clarity
    fig.autofmt_xdate()
    plt.show()

def bayesian_mcmc_simulation(ticker_symbol, year_period = 2):
    time_data, log_price_data = time_price_data(ticker_symbol, year_period = year_period)
    trace = lppl_mcmc(time_data, log_price_data)
    tc_samples, mode_date, hdi_3_date, hdi_97_date = summarize_posterior(trace)

    return tc_samples, mode_date, hdi_3_date, hdi_97_date



if __name__ == "__main__":
    tc_samples, mode_date, hdi_3_date, hdi_97_date = bayesian_mcmc_simulation('^IXIC') # Nasdaq

    print(f"Most Probable Date (Mode): {mode_date.strftime('%Y-%m-%d')}")
    print(f"94% HDI: {hdi_3_date.strftime('%Y-%m-%d')} ~ {hdi_97_date.strftime('%Y-%m-%d')}")
    plot_posterior_tc(tc_samples, mode_date, hdi_3_date, hdi_97_date, '^IXIC')