from flask import Flask, request, jsonify, send_file, render_template
import ccxt
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import io
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator, IchimokuIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator
import seaborn as sns
from datetime import datetime
import warnings
import os
import logging
import pickle
import time
import traceback
from threading import Lock
import urllib.parse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

app = Flask(__name__)

class WaveGrok:
    def __init__(self, exchange_name="kraken"):
        self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        self.markets = {}
        self.valid_symbols = []
        for attempt in range(3):  # Retry up to 3 times
            try:
                self.markets = self.exchange.load_markets(reload=True)
                self.valid_symbols = list(self.markets.keys())
                logging.info(f"Loaded {len(self.valid_symbols)} valid symbols from Kraken: {self.valid_symbols}")
                break  # Success, exit loop
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed to load markets: {str(e)}\n{traceback.format_exc()}")
                if attempt < 2:
                    time.sleep(2)  # Wait before retrying
                else:
                    # Final failure, use fallback
                    self.valid_symbols = [
                        'XBT/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD',
                        'ADA/USD', 'DOT/USD', 'LINK/USD', 'XLM/USD', 'EOS/USD'
                    ]
                    logging.info(f"Using fallback symbols after retries: {self.valid_symbols}")
        self.data = {}
        self.closes = {}
        self.peaks = {}
        self.troughs = {}
        self.rf_model = self._init_rf_model()
        self.lstm_model = self._init_lstm_model()
        self.q_table = {}
        self.trades = []
        self.trade_history = []
        self.portfolio = {"cash": 10000, "assets": {}}
        self.epsilon = 0.3
        self.alpha = 0.1
        self.gamma = 0.9
        self.cache_file = "wavegrok_cache.pkl"
        self.price_cache = {}
        self.cache_lock = Lock()
        self.last_cache_time = {}

    def _normalize_symbol(self, symbol):
        """Normalize symbol to Kraken format."""
        original_symbol = symbol
        symbol = urllib.parse.unquote(symbol).upper()  # Decode %2F to /
        if symbol == "BTC/USDT" or symbol == "BTCUSD":
            symbol = "XBT/USD"
        elif symbol == "XBTUSDT":
            symbol = "XBT/USD"
        logging.debug(f"Normalized symbol: {original_symbol} -> {symbol}")
        return symbol

    def _init_rf_model(self):
        X = np.array([
            [0.02, 0.5, 0.1, 0.05, 0.5, 25, 70, 50, 0.1, 0.01, 0.2, 100, 50, -20, 0.3, 0.4, 10],
            [0.01, -0.4, -0.1, 0.03, 0.3, 15, 30, 40, -0.2, -0.02, 0.1, -50, 20, -10, -0.2, 0.2, -5],
            [0.03, 0.6, 0.2, 0.07, 0.8, 35, 80, 60, 0.3, 0.02, 0.3, 150, 70, 30, 0.4, 0.5, 15],
            [-0.02, -0.3, -0.05, 0.04, 0.4, 20, 20, 30, -0.1, -0.01, 0.15, -80, 40, -15, -0.3, 0.3, -8]
        ])
        y = ["Wave 1", "Wave 2", "Wave 3", "Wave A"]
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
        return model

    def _init_lstm_model(self):
        model = Sequential([LSTM(50, input_shape=(10, 12), return_sequences=False), Dense(8, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        X = np.random.rand(100, 10, 12)
        y = np.random.randint(0, 8, 100)
        model.fit(X, y, epochs=1, verbose=0)
        return model

    def _load_cache(self, symbol, timeframe):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    key = f"{symbol}_{timeframe}"
                    if key in cache and (time.time() - cache[key]['timestamp']) < 300:
                        return cache[key]['data']
        except Exception as e:
            logging.error(f"Cache load failed: {str(e)}")
        return None

    def _save_cache(self, symbol, timeframe, data):
        try:
            cache = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
            key = f"{symbol}_{timeframe}"
            cache[key] = {'data': data, 'timestamp': time.time()}
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            logging.error(f"Cache save failed: {str(e)}")

    def fetch_data(self, symbol, timeframe, limit):
        logging.debug(f"Fetching data for {symbol}, {timeframe}, limit {limit}")
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol not in self.valid_symbols:
            logging.error(f"Invalid symbol: {normalized_symbol}")
            return f"Invalid ticker '{symbol}'. Use a valid Kraken pair from /symbols."
        cached_data = self._load_cache(normalized_symbol, timeframe)
        if cached_data is not None:
            self.data[timeframe] = cached_data
            self.closes[timeframe] = cached_data['close'].values
            return f"Loaded {limit} {timeframe} candles for {normalized_symbol} from cache."
        for attempt in range(3):
            try:
                timeframe_map = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1d'}
                ccxt_timeframe = timeframe_map.get(timeframe.lower(), '1h')
                ohlcv = self.exchange.fetch_ohlcv(normalized_symbol, timeframe=ccxt_timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['momentum'] = df['close'].pct_change()
                df['volume_change'] = df['volume'].pct_change()
                df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
                df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
                df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
                df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
                df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
                df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
                macd = MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_histogram'] = macd.macd_diff()
                df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
                df['psar'] = self._calculate_psar(df['high'], df['low'])
                ichimoku = IchimokuIndicator(df['high'], df['low'])
                df['ichimoku_a'] = ichimoku.ichimoku_a()
                df['ichimoku_b'] = ichimoku.ichimoku_b()
                df['ichimoku_cloud'] = df['ichimoku_a'] - df['ichimoku_b']
                df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
                stoch = StochasticOscillator(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
                df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
                df['roc'] = df['close'].pct_change(periods=12) * 100
                df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
                bb = BollingerBands(df['close'])
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
                df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                df['std_dev'] = df['close'].rolling(window=20).std()
                dc = DonchianChannel(df['high'], df['low'], df['close'])
                df['donchian_upper'] = dc.donchian_channel_hband()
                df['donchian_lower'] = dc.donchian_channel_lband()
                df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
                df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
                df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
                df['cmf'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
                df['force'] = ForceIndexIndicator(df['close'], df['volume']).force_index()
                df['adl'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
                df['adl'] = df['adl'].cumsum().fillna(0)
                df['fractal'] = self._calculate_fractal(df['close'])
                df['fib_236'] = self._calculate_fibonacci(df, 0.236)
                df['fib_382'] = self._calculate_fibonacci(df, 0.382)
                df['fib_618'] = self._calculate_fibonacci(df, 0.618)
                df['pivot_high'] = df['high'].rolling(window=5, center=True).max()
                df['pivot_low'] = df['low'].rolling(window=5, center=True).min()
                df['moon_phase'] = self._get_moon_phase(df.index[-1])
                df.dropna(how='all', inplace=True)
                if df.empty:
                    return f"No valid data for {normalized_symbol} on {timeframe} after cleaning."
                self.data[timeframe] = df
                self.closes[timeframe] = df['close'].values
                self._save_cache(normalized_symbol, timeframe, df)
                return f"Fetched {limit} {timeframe} candles for {normalized_symbol}."
            except ccxt.NetworkError as e:
                logging.error(f"Network error on attempt {attempt + 1} for {normalized_symbol}: {str(e)}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                return f"Network error fetching data: {str(e)}"
            except ccxt.ExchangeError as e:
                logging.error(f"Exchange error for {normalized_symbol}: {str(e)}")
                return f"Exchange error: {str(e)}. Check symbol against /symbols."
            except Exception as e:
                logging.error(f"Unexpected error for {normalized_symbol}: {str(e)}\n{traceback.format_exc()}")
                return f"Error fetching data: {str(e)}"

    def _calculate_fractal(self, closes):
        return (closes.rolling(5).max() - closes.rolling(5).min()) / closes

    def _get_moon_phase(self, dt):
        day = (dt - pd.Timestamp("2025-01-01")).days % 29.53
        return np.abs(np.sin(day / 29.53 * 2 * np.pi))

    def _calculate_psar(self, highs, lows, af_start=0.02, af_max=0.2):
        psar = highs.copy()
        af = af_start
        ep = lows[0]
        trend = 1
        for i in range(1, len(highs)):
            if trend == 1:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if lows[i] < psar[i]:
                    trend = -1
                    psar[i] = ep
                    ep = highs[i]
                    af = af_start
                elif highs[i] > ep:
                    ep = highs[i]
                    af = min(af_max, af + af_start)
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if highs[i] > psar[i]:
                    trend = 1
                    psar[i] = ep
                    ep = lows[i]
                    af = af_start
                elif lows[i] < ep:
                    ep = lows[i]
                    af = min(af_max, af + af_start)
        return psar

    def _calculate_fibonacci(self, df, level):
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        return recent_low + (recent_high - recent_low) * level

    def find_waves(self, timeframe, min_distance=3):
        if timeframe not in self.data:
            return f"No data for {timeframe}—fetch some first!"
        closes = self.data[timeframe]['close'].values
        peaks, _ = find_peaks(closes, distance=min_distance, prominence=closes.std()/10)
        troughs, _ = find_peaks(-closes, distance=min_distance, prominence=closes.std()/10)
        self.peaks[timeframe] = peaks
        self.troughs[timeframe] = troughs
        return f"{timeframe}: Found {len(peaks)} peaks and {len(troughs)} troughs."

    def plot_chart(self, timeframe, indicators=None):
        if timeframe not in self.data:
            logging.error(f"No data for {timeframe}—fetch some first!")
            return None
        df = self.data[timeframe]
        if df.empty:
            logging.error(f"No data to plot for {timeframe}")
            return None
        if indicators is None:
            indicators = ['peaks', 'troughs', 'sma_20', 'sma_50', 'ema_9', 'rsi', 'macd', 'bb', 'fib', 'ichimoku']
        closes = df['close'].values
        peaks, _ = find_peaks(closes, distance=3, prominence=closes.std()/10)
        troughs, _ = find_peaks(-closes, distance=3, prominence=closes.std()/10)
        candle_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        candle_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        peak_data = np.full(len(df), np.nan)
        trough_data = np.full(len(df), np.nan)
        peak_data[peaks] = df['close'].iloc[peaks]
        trough_data[troughs] = df['close'].iloc[troughs]
        apdict = []
        if 'peaks' in indicators:
            apdict.append(mpf.make_addplot(peak_data, type='scatter', markersize=100, marker='x', color='lime', label='Peaks'))
        if 'troughs' in indicators:
            apdict.append(mpf.make_addplot(trough_data, type='scatter', markersize=100, marker='o', color='magenta', label='Troughs'))
        if 'sma_20' in indicators:
            apdict.append(mpf.make_addplot(df['sma_20'], color='cyan', linestyle='--', label='SMA 20'))
        if 'sma_50' in indicators:
            apdict.append(mpf.make_addplot(df['sma_50'], color='yellow', linestyle='--', label='SMA 50'))
        if 'ema_9' in indicators:
            apdict.append(mpf.make_addplot(df['ema_9'], color='green', linestyle='-.', label='EMA 9'))
        if 'bb' in indicators:
            apdict.extend([
                mpf.make_addplot(df['bb_upper'], color='orange', linestyle='--', label='BB Upper'),
                mpf.make_addplot(df['bb_lower'], color='orange', linestyle='--', label='BB Lower'),
            ])
        if 'fib' in indicators:
            apdict.extend([
                mpf.make_addplot(df['fib_236'], color='pink', linestyle='-', label='Fib 23.6%'),
                mpf.make_addplot(df['fib_382'], color='pink', linestyle='-.', label='Fib 38.2%'),
                mpf.make_addplot(df['fib_618'], color='pink', linestyle='--', label='Fib 61.8%'),
            ])
        if 'ichimoku' in indicators:
            apdict.extend([
                mpf.make_addplot(df['ichimoku_a'], color='red', linestyle='-', label='Ichimoku A'),
                mpf.make_addplot(df['ichimoku_b'], color='green', linestyle='-', label='Ichimoku B'),
            ])
        panels = []
        if 'rsi' in indicators:
            panels.append(mpf.make_addplot(df['rsi'], panel=1, color='purple', ylabel='RSI'))
        if 'macd' in indicators:
            panels.extend([
                mpf.make_addplot(df['macd'], panel=2, color='blue', ylabel='MACD'),
                mpf.make_addplot(df['macd_signal'], panel=2, color='orange'),
            ])
        apdict.extend(panels)
        buf = io.BytesIO()
        mpf.plot(
            candle_data,
            type='candle',
            style='yahoo',
            title=f'{timeframe.upper()} Chart',
            ylabel='Price',
            volume=True,
            addplot=apdict,
            figscale=1.5,
            figsize=(12, 9),
            savefig=dict(fname=buf, format='png', bbox_inches='tight', dpi=100)
        )
        logging.info(f"Number of addplot items: {len(apdict)}")
        logging.info(f"Image buffer size after save: {buf.tell()} bytes")
        buf.seek(0)
        logging.info(f"Returning image with size: {buf.tell()} bytes after seek")
        return buf

    def get_meme_hype(self, symbol):
        return random.uniform(0, 1), random.randint(0, 1000)

    def get_q_action(self, state):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = {"Buy": 0, "Sell": 0, "Hold": 0}
        if random.random() < self.epsilon:
            return random.choice(["Buy", "Sell", "Hold"])
        return max(self.q_table[state_str], key=self.q_table[state_str].get)

    def update_q_table(self, state, action, reward, next_state):
        state_str = str(state)
        next_state_str = str(next_state)
        if state_str not in self.q_table:
            self.q_table[state_str] = {"Buy": 0, "Sell": 0, "Hold": 0}
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = {"Buy": 0, "Sell": 0, "Hold": 0}
        old_value = self.q_table[state_str][action]
        next_max = max(self.q_table[next_state_str].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_str][action] = new_value

    def auto_trade(self, symbol, action, price, confidence):
        trade_size = 1000 * (1 + random.uniform(0, 0.5) if confidence > 0.85 else 1)
        if action == "Buy" and self.portfolio["cash"] >= trade_size:
            units = trade_size / price
            self.trades.append({"symbol": symbol, "entry_price": price, "units": units, "time": datetime.now(), "confidence": confidence})
            self.portfolio["cash"] -= trade_size
            if symbol not in self.portfolio["assets"]:
                self.portfolio["assets"][symbol] = 0
            self.portfolio["assets"][symbol] += units
            return f"BUY {symbol} at {price:.2f} - {units:.4f} units for ${trade_size:.2f}", 0
        elif action == "Sell" and self.trades and symbol in self.portfolio["assets"] and self.portfolio["assets"][symbol] > 0:
            trade = self.trades.pop(0)
            units = trade["units"]
            profit = (price - trade["entry_price"]) * units
            self.portfolio["cash"] += price * units
            self.portfolio["assets"][symbol] -= units
            if self.portfolio["assets"][symbol] <= 0:
                del self.portfolio["assets"][symbol]
            self.trade_history.append({"symbol": symbol, "entry_price": trade["entry_price"], "exit_price": price, "profit": profit})
            return f"SELL {symbol} at {price:.2f} - Profit: ${profit:.2f}", 10 if profit > 0 else -5
        return "No trade executed", 0

    def report_performance(self):
        if not self.trade_history:
            return "No trades yet."
        wins = sum(1 for t in self.trade_history if t["profit"] > 0)
        win_rate = wins / len(self.trade_history) * 100
        total_profit = sum(t["profit"] for t in self.trade_history)
        return f"Trades: {len(self.trade_history)}, Win Rate: {win_rate:.2f}%, Profit: ${total_profit:.2f}, Cash: ${self.portfolio['cash']:.2f}"

    def analyze_waves(self, symbol, primary_tf, secondary_tf):
        logging.debug(f"Analyzing waves for {symbol}, {primary_tf}, {secondary_tf}")
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol not in self.valid_symbols:
            return f"Invalid symbol '{symbol}'. Use a valid Kraken pair like '{self.valid_symbols[0]}'."
        if primary_tf not in self.data:
            return f"No data for {primary_tf}—fetch some first!"
        df = self.data[primary_tf]
        closes = self.closes[primary_tf]
        peaks = self.peaks.get(primary_tf, [])
        troughs = self.troughs.get(primary_tf, [])
        if len(peaks) < 2 or len(troughs) < 2:
            return f"Not enough peaks or troughs in {primary_tf}."
        features = [
            df['momentum'].iloc[-1] or 0, df['rsi'].iloc[-1] or 0, df['macd'].iloc[-1] or 0,
            df['bb_width'].iloc[-1] or 0, df['atr'].iloc[-1] or 0, df['adx'].iloc[-1] or 0,
            df['stoch_k'].iloc[-1] or 0, df['cci'].iloc[-1] or 0, df['ichimoku_cloud'].iloc[-1] or 0,
            df['vwap_diff'].iloc[-1] or 0, df['volume_change'].iloc[-1] or 0, df['std_dev'].iloc[-1] or 0,
            df['obv'].diff().iloc[-1] or 0, df['adl'].diff().iloc[-1] or 0, df['williams_r'].iloc[-1] or 0,
            df['cmf'].iloc[-1] or 0, df['force'].iloc[-1] or 0
        ]
        rf_pred = self.rf_model.predict([features])[0]
        lstm_input = np.array(df[['momentum', 'rsi', 'macd', 'bb_width', 'atr', 'adx', 'stoch_k', 'cci',
                                  'ichimoku_cloud', 'vwap_diff', 'close', 'volume_change']].iloc[-10:].fillna(0)).reshape(1, 10, 12)
        lstm_pred = np.argmax(self.lstm_model.predict(lstm_input, verbose=0))
        wave_labels = ["Wave 1", "Wave 2", "Wave 3", "Wave 4", "Wave 5", "Wave A", "Wave B", "Wave C"]
        current_wave = wave_labels[lstm_pred] if random.random() > 0.3 else rf_pred
        direction = "Up" if current_wave in ["Wave 1", "Wave 3", "Wave 5"] else "Down" if current_wave in ["Wave A", "Wave C"] else "Sideways"
        fib_targets = {
            "Buy": df['fib_382'].iloc[-1] if direction == "Up" else df['fib_618'].iloc[-1],
            "Sell": df['fib_618'].iloc[-1] if direction == "Up" else df['fib_382'].iloc[-1],
            "Stop": df['ema_50'].iloc[-1] if direction == "Up" else df['ema_50'].iloc[-1] * 1.02
        }
        signals = {
            "Trend": "Bullish" if df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1] and df['adx'].iloc[-1] > 25 else "Bearish" if df['ema_9'].iloc[-1] < df['ema_21'].iloc[-1] else "Neutral",
            "Momentum": "Overbought" if df['rsi'].iloc[-1] > 70 or df['stoch_k'].iloc[-1] > 80 else "Oversold" if df['rsi'].iloc[-1] < 30 else "Neutral",
            "Volatility": "Breakout" if df['bb_width'].iloc[-1] < df['bb_width'].mean() * 0.5 else "Range",
            "Volume": "Accumulation" if df['obv'].diff().iloc[-1] > 0 else "Distribution"
        }
        sma_cross = "Bullish Crossover" if df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1] and df['sma_50'].iloc[-2] <= df['sma_200'].iloc[-2] else \
                    "Bearish Crossover" if df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1] and df['sma_50'].iloc[-2] >= df['sma_200'].iloc[-2] else "No Crossover"
        state = (current_wave, direction, df['rsi'].iloc[-1] > 70, df['macd'].iloc[-1] > df['macd_signal'].iloc[-1],
                 "Above" if closes[-1] > df['bb_upper'].iloc[-1] else "Below" if closes[-1] < df['bb_lower'].iloc[-1] else "Within",
                 *self.get_meme_hype(normalized_symbol), df['adx'].iloc[-1] > 25)
        action = self.get_q_action(state)
        mtf_confirmed = secondary_tf in self.peaks and len(self.peaks[secondary_tf]) >= 2
        confidence = 0.9 if mtf_confirmed else 0.7
        trade_msg, reward = self.auto_trade(normalized_symbol, action, closes[-1], confidence)
        self.update_q_table(state, action, reward, state)
        self.epsilon = max(0.1, self.epsilon * 0.995)
        rsi_value = df['rsi'].iloc[-1]
        rsi_signal = "Overbought (Sell)" if rsi_value > 70 else "Oversold (Buy)" if rsi_value < 30 else "Neutral"
        atr_value = df['atr'].iloc[-1]
        report = (
            f"WaveGrok Analysis for {normalized_symbol} ({primary_tf})\n"
            f"Current Wave: {current_wave}\n"
            f"Direction: {direction}\n"
            f"Recommended Action: {action}\n"
            f"Buy Point: ${fib_targets['Buy']:.2f}\n"
            f"Sell Target: ${fib_targets['Sell']:.2f}\n"
            f"Stop Loss: ${fib_targets['Stop']:.2f}\n"
            f"Confidence: {confidence:.2%}\n\n"
            f"Trend Indicators:\n"
            f"- SMA Crossover (50/200): {sma_cross}\n"
            f"- SMA 20: {df['sma_20'].iloc[-1]:.2f}\n"
            f"- EMA 9: {df['ema_9'].iloc[-1]:.2f}\n"
            f"- ADX: {df['adx'].iloc[-1]:.2f} ({'Strong Trend' if df['adx'].iloc[-1] > 25 else 'Weak Trend'})\n"
            f"- Ichimoku Cloud: {df['ichimoku_cloud'].iloc[-1]:.2f} ({'Above' if df['close'].iloc[-1] > df['ichimoku_a'].iloc[-1] else 'Below'})\n"
            f"- PSAR: {df['psar'].iloc[-1]:.2f}\n"
            f"Momentum Indicators:\n"
            f"- RSI: {rsi_value:.2f} ({rsi_signal})\n"
            f"- Stochastic %K: {df['stoch_k'].iloc[-1]:.2f} (D: {df['stoch_d'].iloc[-1]:.2f})\n"
            f"- CCI: {df['cci'].iloc[-1]:.2f}\n"
            f"- Williams %R: {df['williams_r'].iloc[-1]:.2f}\n"
            f"Volatility Indicators:\n"
            f"- ATR: {atr_value:.2f}\n"
            f"- Bollinger Width: {df['bb_width'].iloc[-1]:.4f}\n"
            f"- Donchian Upper: {df['donchian_upper'].iloc[-1]:.2f}, Lower: {df['donchian_lower'].iloc[-1]:.2f}\n"
            f"Volume Indicators:\n"
            f"- OBV Change: {df['obv'].diff().iloc[-1]:.2f}\n"
            f"- VWAP Diff: {df['vwap_diff'].iloc[-1]:.4f}\n"
            f"- Chaikin Money Flow: {df['cmf'].iloc[-1]:.4f}\n"
            f"- Force Index: {df['force'].iloc[-1]:.2f}\n"
            f"Support/Resistance:\n"
            f"- Fib 23.6%: {df['fib_236'].iloc[-1]:.2f}\n"
            f"- Fib 38.2%: {df['fib_382'].iloc[-1]:.2f}\n"
            f"- Fib 61.8%: {df['fib_618'].iloc[-1]:.2f}\n"
            f"- Pivot High: {df['pivot_high'].iloc[-1] if not pd.isna(df['pivot_high'].iloc[-1]) else 'N/A'}, "
            f"Low: {df['pivot_low'].iloc[-1] if not pd.isna(df['pivot_low'].iloc[-1]) else 'N/A'}\n"
            f"Extras:\n"
            f"- Moon Phase: {df['moon_phase'].iloc[-1]:.2f}\n\n"
            f"Trade Executed: {trade_msg}\n"
            f"Portfolio Status: {self.report_performance()}"
        )
        return report

agent = WaveGrok()

@app.route('/')
def home():
    logging.debug("Serving home page")
    return render_template('index.html')

@app.route('/fetch', methods=['POST'])
def fetch_data():
    try:
        data = request.get_json()
        logging.debug(f"Received fetch request: {data}")
        if not data:
            logging.error("No JSON data received in /fetch")
            return jsonify({"error": "No data provided"}), 400
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        limit = data.get('limit')
        if not all([symbol, timeframe, limit]):
            logging.error(f"Missing required fields: symbol={symbol}, timeframe={timeframe}, limit={limit}")
            return jsonify({"error": "Missing required fields"}), 400
        limit = int(limit)
        result = agent.fetch_data(symbol, timeframe, limit)
        if "Fetched" in result or "Loaded" in result:
            agent.find_waves(timeframe)
        response = jsonify({"message": result})
        logging.debug(f"Fetch response: {result}")
        return response
    except Exception as e:
        logging.error(f"Error in /fetch: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        logging.debug(f"Received analyze request: {data}")
        if not data:
            logging.error("No JSON data received in /analyze")
            return jsonify({"error": "No data provided"}), 400
        symbol = data.get('symbol')
        primary_tf = data.get('primary_tf')
        secondary_tf = data.get('secondary_tf')
        if not all([symbol, primary_tf, secondary_tf]):
            logging.error(f"Missing required fields: symbol={symbol}, primary_tf={primary_tf}, secondary_tf={secondary_tf}")
            return jsonify({"error": "Missing required fields"}), 400
        result = agent.analyze_waves(symbol, primary_tf, secondary_tf)
        response = jsonify({"message": result})
        logging.debug(f"Analyze response: {result}")
        return response
    except Exception as e:
        logging.error(f"Error in /analyze: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/chart/<timeframe>')
def get_chart(timeframe):
    try:
        indicators = request.args.get('indicators', 'peaks,troughs,sma_20,sma_50,ema_9,rsi,macd,bb,fib,ichimoku').split(',')
        logging.debug(f"Generating chart for {timeframe} with indicators: {indicators}")
        img = agent.plot_chart(timeframe, indicators)
        if img is None:
            logging.error(f"Chart generation failed for {timeframe}")
            return jsonify({"error": "No data"}), 400
        return send_file(img, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error in /chart: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/price/<path:symbol>')
def get_price(symbol):
    try:
        logging.info(f"Received request for /price/{symbol}")
        normalized_symbol = agent._normalize_symbol(symbol)
        with agent.cache_lock:
            if normalized_symbol in agent.price_cache and (time.time() - agent.last_cache_time.get(normalized_symbol, 0)) < 10:
                logging.debug(f"Returning cached price for {normalized_symbol}")
                return jsonify({"price": agent.price_cache[normalized_symbol]})
        if normalized_symbol not in agent.valid_symbols:
            example_symbol = agent.valid_symbols[0] if agent.valid_symbols else 'XBT/USD'
            logging.error(f"Invalid symbol: {normalized_symbol}. Valid symbols: {agent.valid_symbols[:5]}...")
            return jsonify({"error": f"Invalid symbol '{symbol}'. Use a Kraken pair like '{example_symbol}'."}), 400
        ticker = agent.exchange.fetch_ticker(normalized_symbol)
        price = ticker.get('last', None)
        if price is None:
            logging.warning(f"No 'last' price in ticker for {normalized_symbol}: {ticker}")
            return jsonify({"error": "No price data available"}), 404
        with agent.cache_lock:
            agent.price_cache[normalized_symbol] = price
            agent.last_cache_time[normalized_symbol] = time.time()
        logging.info(f"Fetched price for {normalized_symbol}: ${price:.2f}")
        return jsonify({"price": price})
    except ccxt.NetworkError as e:
        logging.error(f"Network error fetching price for {symbol}: {str(e)}")
        return jsonify({"error": "Network issue - try again later"}), 503
    except ccxt.ExchangeError as e:
        logging.error(f"Exchange error fetching price for {symbol}: {str(e)}")
        return jsonify({"error": f"Exchange error: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Unexpected error fetching price for {symbol}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Price unavailable"}), 500

@app.route('/symbols')
def get_symbols():
    try:
        logging.info("Received request for /symbols")
        symbols = list(agent.markets.keys())
        logging.debug(f"Returning {len(symbols)} symbols")
        return jsonify({"symbols": symbols})
    except Exception as e:
        logging.error(f"Error fetching symbols: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Unable to fetch symbols"}), 500

@app.route('/sentiment/<path:symbol>')
def get_sentiment(symbol):
    try:
        logging.info(f"Received request for /sentiment/{symbol}")
        normalized_symbol = agent._normalize_symbol(symbol)
        if normalized_symbol not in agent.valid_symbols:
            example_symbol = agent.valid_symbols[0] if agent.valid_symbols else 'XBT/USD'
            logging.error(f"Invalid symbol: {normalized_symbol}. Valid symbols: {agent.valid_symbols[:5]}...")
            return jsonify({"error": f"Invalid symbol '{symbol}'. Use a Kraken pair like '{example_symbol}'."}), 400
        base = agent.markets[normalized_symbol]['base']
        if base.startswith('X'):
            base = base[1:]
        search_term = f"${base}"
        posts = []  # Placeholder for future X integration
        sentiment_score = random.uniform(-1, 1)
        sentiment = "Bullish" if sentiment_score > 0.2 else "Bearish" if sentiment_score < -0.2 else "Neutral"
        logging.info(f"Sentiment for {normalized_symbol}: {sentiment} (score: {sentiment_score:.2f})")
        return jsonify({"sentiment": sentiment, "score": sentiment_score})
    except Exception as e:
        logging.error(f"Error fetching sentiment for {symbol}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Sentiment unavailable"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)