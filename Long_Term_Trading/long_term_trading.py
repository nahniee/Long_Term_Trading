import pandas as pd
from tqdm import tqdm
import yfinance as yf

from dataclasses import dataclass, field
from config import EnvConfig
from llm_api_model import OpenAIModel
from lppl_simulation import bayesian_mcmc_simulation, plot_posterior_tc
from stats_model_process import get_gbm_path_simulation, get_gbm_drift_calculation
from clam_inference import CONFIG as CLAM_CONFIG, load_prediction_tools, predict_single_ticker

def predict_market_crash():
    """run LPPL (Log-Periodic Power Law) analysis on Nasdaq(^IXIC) and S&P500(^GSPC) to predict market crash"""
    try:
        tc_samples, mode_date, hdi_3_date, hdi_97_date = bayesian_mcmc_simulation('^IXIC') # Nasdaq

        print(f"Most Probable Date (Mode): {mode_date.strftime('%Y-%m-%d')}")
        print(f"94% HDI: {hdi_3_date.strftime('%Y-%m-%d')} ~ {hdi_97_date.strftime('%Y-%m-%d')}")
        plot_posterior_tc(tc_samples, mode_date, hdi_3_date, hdi_97_date, '^IXIC')
    except Exception as e:
        print(f"Error running LPPL analysis for ^IXIC: {e}")

    try:
        tc_samples, mode_date, hdi_3_date, hdi_97_date = bayesian_mcmc_simulation('^GSPC') # S&P500
        print(f"Most Probable Date (Mode): {mode_date.strftime('%Y-%m-%d')}")
        print(f"94% HDI: {hdi_3_date.strftime('%Y-%m-%d')} ~ {hdi_97_date.strftime('%Y-%m-%d')}")
        plot_posterior_tc(tc_samples, mode_date, hdi_3_date, hdi_97_date, '^GSPC')
    except Exception as e:
        print(f"Error running LPPL analysis for ^GSPC: {e}")


# Interesting tickers for long-term trading
POSSIBLE_TRADING_TICKERS = [
    "PLTR", "AMD", "IONQ", "NVDA", "AAPL",
    "TSLA", "GOOGL", "AMZN", "META", "NFLX",
    "SMR", "RKLB", "LCID", "IREN", "FLEX",
    "SMCI", "RIVN", "DIS", "KO", "WMT",
    "DLR", "VRT", "TLN", "IOT", "SOFI",
    "QQQ", "SPY", "VIX", "BTC-USD", "ETH-USD",
]

@dataclass
class TickerData:
    ticker_name: str
    sentiment_counts: dict = field(default_factory=lambda: {"positive": 0, "negative": 0, "neutral": 0})
    gbm_path: float = 0.0
    clam_prediction_growth: float = 0.0

class LongTermPreprocessTrading:
    def __init__(self, env_config):
        self.env_config = env_config
        self.llm_model = OpenAIModel(env_config.openai_key)
        self.possible_trading_tickers = [TickerData(ticker) for ticker in POSSIBLE_TRADING_TICKERS]

    def create_news_sentiment_df(self):
        """Create a DataFrame for news sentiment analysis results"""
        data = []
        for ticker in self.possible_trading_tickers:
            sentiment = ticker.sentiment_counts
            total_articles = sum(sentiment.values())
            sentiment_score = (sentiment["positive"] - sentiment["negative"]) / max(total_articles, 1)
            
            data.append({
                'Rank': 0,  # Will be filled after sorting
                'Ticker': ticker.ticker_name,
                'Positive': sentiment["positive"],
                'Negative': sentiment["negative"],
                'Neutral': sentiment["neutral"],
                'Total Articles': total_articles,
                'Sentiment Score': f"{sentiment_score:.3f}"
            })
        
        # Sort by sentiment score and add rank
        df = pd.DataFrame(data)
        df = df.sort_values('Sentiment Score', ascending=False, key=lambda x: pd.to_numeric(x)).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        df = df[['Rank', 'Ticker', 'Positive', 'Negative', 'Neutral', 'Total Articles', 'Sentiment Score']]
        return df

    def create_gbm_results_df(self):
        """Create a DataFrame for GBM path simulation results"""
        data = []
        for ticker in self.possible_trading_tickers:
            data.append({
                'Rank': 0,  # Will be filled after sorting
                'Ticker': ticker.ticker_name,
                'GBM Score': f"{ticker.gbm_path:.6f}",
                'GBM Score (Raw)': ticker.gbm_path
            })
        
        # Sort by GBM score and add rank
        df = pd.DataFrame(data)
        df = df.sort_values('GBM Score (Raw)', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Remove raw column and reorder
        df = df[['Rank', 'Ticker', 'GBM Score']]
        return df

    def create_clam_results_df(self):
        """Create a DataFrame for CLAM inference results"""
        data = []
        for ticker in self.possible_trading_tickers:
            data.append({
                'Rank': 0,  # Will be filled after sorting
                'Ticker': ticker.ticker_name,
                'Predicted Growth': f"{ticker.clam_prediction_growth:.4f}",
                'Growth %': f"{ticker.clam_prediction_growth * 100:.2f}%",
                'Growth (Raw)': ticker.clam_prediction_growth
            })
        
        # Sort by predicted growth and add rank
        df = pd.DataFrame(data)
        df = df.sort_values('Growth (Raw)', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Remove raw column and reorder
        df = df[['Rank', 'Ticker', 'Predicted Growth', 'Growth %']]
        return df

    def run_news_sentiment_analysis(self):
        print("Running news sentiment analysis...")
        for ticker in tqdm(self.possible_trading_tickers, desc="Analyzing news sentiment"):
            try:
                # Get news articles for the ticker
                articles = yf.Ticker(ticker.ticker_name).news

                # query LLM to analyze sentiment
                for article in articles:
                    prompt = (
                        f"Analyze the sentiment of the following {ticker.ticker_name} article:\n"
                        f"Title: {article['content']['title']}\n"
                        f"Summary: {article['content']['summary']}\n"
                        "Return the sentiment as 'positive', 'negative', or 'neutral' at the end"
                    )
                    response = self.llm_model.query_model(prompt)

                    # get "positive", "negative", or "neutral" at the end
                    response = response.strip().splitlines()[-1].lower()
                    if "positive" in response:
                        ticker.sentiment_counts["positive"] += 1
                    elif "negative" in response:
                        ticker.sentiment_counts["negative"] += 1
                    else:
                        ticker.sentiment_counts["neutral"] += 1
            except Exception as e:
                print(f"Error analyzing news sentiment for {ticker.ticker_name}: {e}")
                continue
        
        # Create and return DataFrame
        df = self.create_news_sentiment_df()
        return df

    def run_gbm_path(self):
        print("Running GBM path simulation...")
        for ticker in tqdm(self.possible_trading_tickers, desc="GBM Analysis"):
            try:
                gbm_normalized_score = get_gbm_path_simulation(self.env_config, ticker.ticker_name, mode="quarterly")
                ticker.gbm_path = float(gbm_normalized_score)
            except Exception as e:
                print(f"Error running GBM path simulation for {ticker.ticker_name}: {e}")
                ticker.gbm_path = 0.0
                continue

        # Create and return DataFrame
        df = self.create_gbm_results_df()
        return df

    def run_clam_inference(self):
        print("Running CLAM inference...")
        config = CLAM_CONFIG['quarterly']
        model, scaler = load_prediction_tools(config)
        prediction_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        if model and scaler:
            for ticker in tqdm(self.possible_trading_tickers, desc="CLAM Inference"):
                try:
                    result = predict_single_ticker(ticker.ticker_name, model, scaler, config, prediction_date)
                    if result is None:
                        print(f"Prediction failed or insufficient data for {ticker.ticker_name}.")
                        ticker.clam_prediction_growth = 0.0
                        continue
                    else:
                        ticker.clam_prediction_growth = result['growth_rate']
                except Exception as e:
                    print(f"Error in CLAM prediction for {ticker.ticker_name}: {e}")
                    ticker.clam_prediction_growth = 0.0

        # Create and return DataFrame
        df = self.create_clam_results_df()
        return df

    def print_results_summary(self, news_df, gbm_df, clam_df):
        """Print a combined summary of top performers"""
        print("\n" + "="*80)
        print("📊 LONG-TERM TRADING ANALYSIS SUMMARY")
        print("="*80)
        
        # Top 25 from each analysis
        print("\n🗞️  TOP 25 NEWS SENTIMENT ANALYSIS")
        print("-" * 50)
        print(news_df.head(25).to_string(index=False))

        print("\n📈 TOP 25 GBM PATH SIMULATION")
        print("-" * 50)
        print(gbm_df.head(25).to_string(index=False))

        print("\n🤖 TOP 25 CLAM ML PREDICTION")
        print("-" * 50)
        print(clam_df.head(25).to_string(index=False))

        # Combined scoring (simple average of ranks)
        # combined_data = []
        # for ticker_name in POSSIBLE_TRADING_TICKERS:
        #     news_rank = news_df[news_df['Ticker'] == ticker_name]['Rank'].iloc[0] if len(news_df[news_df['Ticker'] == ticker_name]) > 0 else len(POSSIBLE_TRADING_TICKERS)
        #     gbm_rank = gbm_df[gbm_df['Ticker'] == ticker_name]['Rank'].iloc[0] if len(gbm_df[gbm_df['Ticker'] == ticker_name]) > 0 else len(POSSIBLE_TRADING_TICKERS)
        #     clam_rank = clam_df[clam_df['Ticker'] == ticker_name]['Rank'].iloc[0] if len(clam_df[clam_df['Ticker'] == ticker_name]) > 0 else len(POSSIBLE_TRADING_TICKERS)
            
        #     avg_rank = (news_rank + gbm_rank + clam_rank) / 3
        #     combined_data.append({
        #         'Ticker': ticker_name,
        #         'News Rank': news_rank,
        #         'GBM Rank': gbm_rank,
        #         'CLAM Rank': clam_rank,
        #         'Average Rank': f"{avg_rank:.1f}"
        #     })
        
        # combined_df = pd.DataFrame(combined_data)
        # combined_df = combined_df.sort_values('Average Rank', key=lambda x: pd.to_numeric(x)).reset_index(drop=True)
        # combined_df['Final Rank'] = range(1, len(combined_df) + 1)
        # combined_df = combined_df[['Final Rank', 'Ticker', 'News Rank', 'GBM Rank', 'CLAM Rank', 'Average Rank']]
        
        # print("\n🏆 COMBINED RANKING (Top 10)")
        # print("-" * 60)
        # print(combined_df.head(10).to_string(index=False))
        
        # return combined_df

    def run_technical_analysis(self):
        print("Starting comprehensive technical analysis...")
        
        # Run individual analyses
        news_result = self.run_news_sentiment_analysis()
        gbm_result = self.run_gbm_path()
        clam_result = self.run_clam_inference()
        
        # Print detailed results and summary
        self.print_results_summary(news_result, gbm_result, clam_result)
        
        # Return top combined performers
        # top_performers = combined_ranking.head(10)['Ticker'].tolist()
        # print(f"\n✅ Analysis complete! Top 10 recommended tickers: {top_performers}")
        
        # return top_performers
        
if __name__ == "__main__":
    import pyprojroot
    import os

    root_dir = pyprojroot.find_root(pyprojroot.has_dir('.git'))
    env_path = os.path.join(root_dir, ".env")
    env_config = EnvConfig(env_path)
    
    long_term_preprocess = LongTermPreprocessTrading(env_config)
    selected_tickers = long_term_preprocess.run_technical_analysis()

    predict_market_crash()