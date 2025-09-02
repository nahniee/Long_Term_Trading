# train_model.py
import time
import datetime
import joblib

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, Layer, RepeatVector, TimeDistributed, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from config import EnvConfig

"""  
Model configuration guidelines:

* Long term (Quarterly): 
  - Training Frequency: Weekly (e.g., every Sunday night)
  - SEQ_LENGTH = 252 (about 1 trading year of input)
  - FORECAST_DAYS = 65 (approx. 3 months prediction horizon)
  - CNN: 3 layers, 128 filters, kernel_size = 5
  - LSTM: 3 layers, 256 units

* Short term (Hourly): 
  - Training Frequency: Daily (e.g., every night after market close)
  - SEQ_LENGTH = 140 (about 140 hours (20 days) of input)
  - FORECAST_STEPS = 7 (next 7 hours prediction)
  - CNN: 2 layers, 64 filters, kernel_size = 3
  - LSTM: 2 layers, 128 units
"""
# Configuration for model training and prediction
CONFIG = {
    'quarterly': {
        'seq_length': 252,
        'forecast_horizon': 65,
        'interval': '1d', # Daily data
        'cnn_layers': [
            {'filters': 128, 'kernel_size': 5},
            {'filters': 128, 'kernel_size': 5},
            {'filters': 128, 'kernel_size': 5}
        ],
        'lstm_layers': [
            {'units': 256},
            {'units': 256},
            {'units': 256}
        ],
        'data_start_date': '2010-01-01',
        'validation_period': {'years': 2}
    },
    'hourly': {
        'seq_length': 140,
        'forecast_horizon': 7,
        'interval': '1h', # Hourly data
        'cnn_layers': [
            {'filters': 64, 'kernel_size': 3},
            {'filters': 64, 'kernel_size': 3}
        ],
        'lstm_layers': [
            {'units': 128},
            {'units': 128}
        ],
        # yfinance provides max 730 days of hourly data
        'data_start_date': (datetime.date.today() - datetime.timedelta(days=729)).strftime('%Y-%m-%d'),
        'validation_period': {'months': 3}
    }
}
FEATURE_COUNT = 5 # Open, High, Low, Close, Volume

# Custom Attention layer and custom metric (Directional Accuracy)
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs): super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight matrix for attention scoring
        self.W = self.add_weight(name='att_w', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True) 
        # Bias term
        self.b = self.add_weight(name='att_b', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Attention scores
        e = K.tanh(K.dot(x, self.W) + self.b)
        alpha = K.softmax(K.squeeze(e, axis=-1))
        # Weighted sum to generate context vector
        context = K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)
        return context

# Measures accuracy of predicted price movement direction
def directional_accuracy(y_true, y_pred):
    # # Compare the sign of predicted vs actual Close price changes
    # true_direction = K.sign(y_true[:, :, 3]) # Close is the 4th feature (index 3)
    # pred_direction = K.sign(y_pred[:, :, 3])

    # Compare the sign of predicted vs actual High price changes
    true_direction = K.sign(y_true[:, :, 1]) # High is the 2nd feature (index 1)
    pred_direction = K.sign(y_pred[:, :, 1])

    correct_direction = K.equal(true_direction, pred_direction)
    return K.mean(tf.cast(correct_direction, tf.float32))

# Model creation function
def create_model(config):
    seq_len = config['seq_length']
    forecast_horizon = config['forecast_horizon']
    
    encoder_inputs = Input(shape=(seq_len, FEATURE_COUNT))
    x = encoder_inputs

    # Build CNN layers dynamically
    for layer_params in config['cnn_layers']:
        x = Conv1D(filters=layer_params['filters'], kernel_size=layer_params['kernel_size'], padding='causal', activation='relu')(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

    # Build LSTM layers dynamically
    for layer_params in config['lstm_layers']:
        x = LSTM(units=layer_params['units'], return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
    
    # Attention context vector
    context_vector = Attention()(x)

    # Decoder for sequence forecasting
    decoder_lstm_units = config['lstm_layers'][-1]['units']
    decoder_inputs = RepeatVector(forecast_horizon)(context_vector)
    decoder_lstm = LSTM(decoder_lstm_units, return_sequences=True)(decoder_inputs)
    outputs = TimeDistributed(Dense(FEATURE_COUNT))(decoder_lstm)
    
    model = Model(encoder_inputs, outputs)

    # Compile with custom metric
    # The Huber loss function and the AdamW optimizer are the best combination for stock data with high volatility and many unpredictable outliers
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4), 
                  loss='huber', 
                  metrics=[directional_accuracy])
    return model

# Sliding window sequence generation
def create_sequences(data, seq_len, forecast_len):
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_len + 1):
        X.append(data[i:(i + seq_len)])
        y.append(data[(i + seq_len):(i + seq_len + forecast_len)])
    return np.array(X), np.array(y)

# Main training loop
def clam_train(env_config, model_type):
    print(f"Starting training for [{model_type.upper()}] model")
    
    # Load the correct configuration
    config = CONFIG[model_type]
    seq_length = config['seq_length']
    forecast_horizon = config['forecast_horizon']
    
    # Training tickers (multi-sector, 100+ tickers)
    TRAINING_TICKERS = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 
                        'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 
                        'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 
                        'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAPL', 'APTV', 'ACGL', 'ADM', 
                        'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 
                        'BKR', 'BALL', 'BAC', 'BBWI', 'BAX', 'BDX', 'BBY', 'BIO', 'TECH', 'BIIB', 
                        'BLK', 'BX', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 
                        'BLDR', 'BG', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CAT', 
                        'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CHRW', 'CRL', 'SCHW', 
                        'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 
                        'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 
                        'STZ', 'COO', 'CPRT', 'GLW', 'CSGP', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 
                        'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 
                        'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 
                        'ECL', 'EIX', 'EW', 'EA', 'ELV', 'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 
                        'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EG', 'EVRG', 'ES', 'EXC', 
                        'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FSLR', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 
                        'FICO', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'BEN', 'FCX', 'GRMN', 'IT', 'GEN', 'GNRC', 
                        'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'HAL', 'HIG', 'HAS', 
                        'HCA', 'HSIC', 'HSY', 'HES', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 
                        'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'PODD', 'INTC', 
                        'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'IQV', 'IRM', 'JBHT', 'JBL', 
                        'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KDP', 'KEY', 'KEYS', 'KMB', 
                        'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LVS', 'LDOS', 'LEN', 
                        'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 
                        'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 
                        'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'MOH', 'TAP', 
                        'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 
                        'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 
                        'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 
                        'OKE', 'ORCL', 'PCAR', 'PKG', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 
                        'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 
                        'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'QRVO', 'PWR', 
                        'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 
                        'RVTY', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 
                        'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SNA', 'SO', 'LUV', 'SWKS', 'SBUX', 
                        'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 
                        'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 
                        'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 
                        'TSN', 'USB', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 
                        'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VMC', 
                        'WRB', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 
                        'WST', 'WDC', 'WY', 'WHR', 'WMB', 'WTW', 'GWW', 'WYNN', 'XEL', 'XYL', 
                        'YUM', 'ZBRA', 'ZBH', 'ZTS']

    # Data download and preprocessing
    all_processed_dfs = []
    failed_tickers = []
    
    for ticker in tqdm(TRAINING_TICKERS, desc="Processing Tickers"):
        try:
            time.sleep(1)  # To respect API rate limits
            end_date = pd.Timestamp.now()
            raw_df = yf.download(ticker, 
                                 start=config['data_start_date'], 
                                 end=end_date, 
                                 interval=config['interval'],
                                 progress=False,
                                 auto_adjust=False)
    
            if raw_df.empty:
                failed_tickers.append(ticker)
                continue

            processed_df = pd.DataFrame(index=raw_df.index)
            processed_df['Open'] = np.log(raw_df['Open']).diff()
            processed_df['High'] = np.log(raw_df['High']).diff()
            processed_df['Low'] = np.log(raw_df['Low']).diff()
            processed_df['Close'] = np.log(raw_df['Close']).diff()
            processed_df['Volume'] = np.log1p(raw_df['Volume']).diff()

            # Drop initial NaN rows caused by diff()
            processed_df.dropna(inplace=True)
            all_processed_dfs.append(processed_df)
        except Exception as e:
            failed_tickers.append(ticker)

    if not failed_tickers: 
        print("All tickers processed successfully!")
    else:
        print(f"Failed to process: {failed_tickers}")
        return False

    # Merge all tickers and sort by date
    full_processed_df = pd.concat(all_processed_dfs).sort_index()

    # Data splitting
    last_date = full_processed_df.index.max()
    split_date = last_date - pd.DateOffset(**config['validation_period'])
    train_df = full_processed_df[full_processed_df.index < split_date]
    val_df = full_processed_df[full_processed_df.index >= split_date]
    
    print(f"Training data period: {train_df.index.min()} ~ {train_df.index.max()}")
    print(f"Validation data period: {val_df.index.min()} ~ {val_df.index.max()}")

    # Scaling and Sequencing
    scaler = MinMaxScaler(feature_range=(-1, 1))    # Use (-1,1) since returns can be negative
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    
    # Generate sequences
    X_train, y_train = create_sequences(train_scaled, seq_length, forecast_horizon)
    X_val, y_val = create_sequences(val_scaled, seq_length, forecast_horizon)

    # Build and train model
    model = create_model(config)
    model.summary()
    
    callbacks = [
        # Save the best model based on directional accuracy
        EarlyStopping(monitor='val_directional_accuracy', mode='max', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
    ]
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)

    # Save Artifacts
    model_filename = f"./ml_model/{model_type}_model.h5"
    scaler_filename = f"./ml_model/{model_type}_scaler.pkl"
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"\nTraining complete. {model_filename} and {scaler_filename} have been saved.")
    return True

if __name__ == '__main__':
    import os
    """
    python -m src.ticker_data_processor.clam_train
    """
    current_file_pwd = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file_pwd))
    parent_dir = os.path.dirname(src_dir)
    env_path = os.path.join(parent_dir, ".env")
    env_config = EnvConfig(env_path)

    # To train a different model, options are 'quarterly' or 'hourly'
    model_to_train = 'hourly'
    clam_train(env_config, model_to_train)