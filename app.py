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

warnings.filterwarnings("ignore")

app = Flask(__name__)

class WaveGrok:
    def __init__(self, exchange_name="kraken"):
        self.exchange = getattr(ccxt, exchange_name)()
        self.markets = self.exchange.load_markets()
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

    def _init_rf_model(self):
        X = np.array([[0.02, 0.5, 0.1, 60, 0.5, 0.9, 0.5, 25, 70, 100, 0.5, 0.1],
                      [0.01, -0.4, 0.05, 40, -0.3, 0.8, 0.3, 15, 30, -50, -0.2, -0.05],
                      [0.03, 0.6, 0.2, 70, 0.8, 1.1, 0.7, 35, 80, 150, 0.8, 0.15],
                      [-0.02, -0.3, 0.1, 30, -0.5, 0.7, 0.4, 20, 20, -80, -0.3, -0.08]])
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

    def fetch_data(self, symbol, timeframe, limit):
        if symbol not in self.markets:
            return f"Invalid ticker '{symbol}' for {self.exchange.name}."
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Core Price Metrics
            df['momentum'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()

            # Trend Indicators
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

            # Momentum Indicators
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
            df['roc'] = df['close'].pct_change(periods=12) * 100
            df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

            # Volatility Indicators
            bb = BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['std_dev'] = df['close'].rolling(window=20).std()
            dc = DonchianChannel(df['high'], df['low'], df['close'])
            df['donchian_upper'] = dc.donchian_channel_hband()
            df['donchian_lower'] = dc.donchian_channel_lband()

            # Volume Indicators
            df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
            df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
            df['cmf'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            df['force'] = ForceIndexIndicator(df['close'], df['volume']).force_index()
            df['adl'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
            df['adl'] = df['adl'].cumsum().fillna(0)

            # Support/Resistance and Waves
            df['fractal'] = self._calculate_fractal(df['close'])
            df['fib_236'] = self._calculate_fibonacci(df, 0.236)
            df['fib_382'] = self._calculate_fibonacci(df, 0.382)
            df['fib_618'] = self._calculate_fibonacci(df, 0.618)
            df['pivot_high'] = df['high'].rolling(window=5, center=True).max()
            df['pivot_low'] = df['low'].rolling(window=5, center=True).min()

            # Extras
            df['moon_phase'] = self._get_moon_phase(df['timestamp'].iloc[-1])

            self.data[timeframe] = df
            self.closes[timeframe] = df['close'].values
            return f"Fetched {limit} {timeframe} candles for {symbol}."
        except Exception as e:
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
        if timeframe not in self.closes:
            return f"No data for {timeframe}—fetch some first!"
        closes = self.closes[timeframe]
        self.peaks[timeframe], _ = find_peaks(closes, distance=min_distance, prominence=closes.std()/10)
        self.troughs[timeframe], _ = find_peaks(-closes, distance=min_distance, prominence=closes.std()/10)
        return f"{timeframe}: Found {len(self.peaks[timeframe])} peaks and {len(self.troughs[timeframe])} troughs."

    def plot_chart(self, timeframe):
        if timeframe not in self.data:
            return None
        df = self.data[timeframe]
        closes = self.closes[timeframe]
        fig = plt.figure(figsize=(12, 12))
        
        # Subplots: Candlestick, RSI, MACD, ATR
        ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=6)  # Candlestick
        ax2 = plt.subplot2grid((12, 1), (6, 0), rowspan=2)  # RSI
        ax3 = plt.subplot2grid((12, 1), (8, 0), rowspan=2)  # MACD
        ax4 = plt.subplot2grid((12, 1), (10, 0), rowspan=2) # ATR

        # Candlestick Chart
        candle_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')
        mpf.plot(candle_data, type='candle', ax=ax1, style='charles', title=f"WaveGrok - {timeframe}")
        ax1.plot(self.peaks[timeframe], closes[self.peaks[timeframe]], "x", label='Peaks', color='lime')
        ax1.plot(self.troughs[timeframe], closes[self.troughs[timeframe]], "o", label='Troughs', color='magenta')
        ax1.plot(df['sma_50'], label='SMA 50', color='yellow', linestyle='--')
        ax1.plot(df['sma_200'], label='SMA 200', color='red', linestyle='--')
        ax1.legend()

        # RSI
        rsi_value = df['rsi'].iloc[-1]
        ax2.plot(df['rsi'], label='RSI', color='purple')
        ax2.axhline(70, ls='--', color='red', label='Overbought (70)')
        ax2.axhline(30, ls='--', color='green', label='Oversold (30)')
        ax2.set_title(f"RSI (14): {rsi_value:.2f}")
        ax2.legend()

        # MACD
        ax3.plot(df['macd'], label='MACD', color='blue')
        ax3.plot(df['macd_signal'], label='Signal', color='orange')
        ax3.bar(df.index, df['macd_histogram'], label='Histogram', color='gray', alpha=0.5)
        ax3.legend()

        # ATR
        atr_value = df['atr'].iloc[-1]
        ax4.plot(df['atr'], label='ATR', color='orange')
        ax4.set_title(f"ATR (14): {atr_value:.2f}")
        ax4.legend()

        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        return img

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
        return (f"Trades: {len(self.trade_history)}, Win Rate: {win_rate:.2f}%, "
                f"Profit: ${total_profit:.2f}, Cash: ${self.portfolio['cash']:.2f}")

    def analyze_waves(self, symbol, primary_tf, secondary_tf):
        if primary_tf not in self.data:
            return f"No data for {primary_tf}—fetch some first!"
        df = self.data[primary_tf]
        closes = self.closes[primary_tf]
        peaks = self.peaks[primary_tf]
        troughs = self.troughs[primary_tf]
        if len(peaks) < 2 or len(troughs) < 2:
            return f"Not enough peaks or troughs in {primary_tf}."

        # Feature vector for ML models
        features = [
            df['momentum'].iloc[-1] or 0, df['rsi'].iloc[-1] or 0, df['macd'].iloc[-1] or 0,
            df['bb_width'].iloc[-1] or 0, df['atr'].iloc[-1] or 0, df['adx'].iloc[-1] or 0,
            df['stoch_k'].iloc[-1] or 0, df['cci'].iloc[-1] or 0, df['ichimoku_cloud'].iloc[-1] or 0,
            df['vwap_diff'].iloc[-1] or 0, df['volume_change'].iloc[-1] or 0, df['std_dev'].iloc[-1] or 0,
            df['obv'].diff().iloc[-1] or 0, df['adl'].diff().iloc[-1] or 0, df['williams_r'].iloc[-1] or 0,
            df['cmf'].iloc[-1] or 0, df['force'].iloc[-1] or 0
        ]

        # Elliott Wave Prediction
        rf_pred = self.rf_model.predict([features])[0]
        lstm_input = np.array(df[['momentum', 'rsi', 'macd', 'bb_width', 'atr', 'adx', 'stoch_k', 'cci',
                                  'ichimoku_cloud', 'vwap_diff', 'close', 'volume_change']].iloc[-10:].fillna(0)).reshape(1, 10, 12)
        lstm_pred = np.argmax(self.lstm_model.predict(lstm_input, verbose=0))
        wave_labels = ["Wave 1", "Wave 2", "Wave 3", "Wave 4", "Wave 5", "Wave A", "Wave B", "Wave C"]
        current_wave = wave_labels[lstm_pred] if random.random() > 0.3 else rf_pred

        # Direction and Targets
        direction = "Up" if current_wave in ["Wave 1", "Wave 3", "Wave 5"] else "Down" if current_wave in ["Wave A", "Wave C"] else "Sideways"
        fib_targets = {
            "Buy": df['fib_382'].iloc[-1] if direction == "Up" else df['fib_618'].iloc[-1],
            "Sell": df['fib_618'].iloc[-1] if direction == "Up" else df['fib_382'].iloc[-1],
            "Stop": df['ema_50'].iloc[-1] if direction == "Up" else df['ema_50'].iloc[-1] * 1.02
        }

        # Confluence Signals
        signals = {
            "Trend": "Bullish" if df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1] and df['adx'].iloc[-1] > 25 else "Bearish" if df['ema_9'].iloc[-1] < df['ema_21'].iloc[-1] else "Neutral",
            "Momentum": "Overbought" if df['rsi'].iloc[-1] > 70 or df['stoch_k'].iloc[-1] > 80 else "Oversold" if df['rsi'].iloc[-1] < 30 else "Neutral",
            "Volatility": "Breakout" if df['bb_width'].iloc[-1] < df['bb_width'].mean() * 0.5 else "Range",
            "Volume": "Accumulation" if df['obv'].diff().iloc[-1] > 0 else "Distribution"
        }

        # SMA Crossover Signal
        sma_cross = "Bullish Crossover" if df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1] and df['sma_50'].iloc[-2] <= df['sma_200'].iloc[-2] else \
                    "Bearish Crossover" if df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1] and df['sma_50'].iloc[-2] >= df['sma_200'].iloc[-2] else "No Crossover"

        # Q-Learning Decision
        state = (current_wave, direction, df['rsi'].iloc[-1] > 70, df['macd'].iloc[-1] > df['macd_signal'].iloc[-1],
                 "Above" if closes[-1] > df['bb_upper'].iloc[-1] else "Below" if closes[-1] < df['bb_lower'].iloc[-1] else "Within",
                 *self.get_meme_hype(symbol), df['adx'].iloc[-1] > 25)
        action = self.get_q_action(state)
        mtf_confirmed = secondary_tf in self.peaks and len(self.peaks[secondary_tf]) >= 2
        confidence = 0.9 if mtf_confirmed else 0.7
        trade_msg, reward = self.auto_trade(symbol, action, closes[-1], confidence)
        self.update_q_table(state, action, reward, state)
        self.epsilon = max(0.1, self.epsilon * 0.995)

        # Mega Report with Adjusted Indicators
        rsi_value = df['rsi'].iloc[-1]
        rsi_signal = "Overbought (Sell)" if rsi_value > 70 else "Oversold (Buy)" if rsi_value < 30 else "Neutral"
        atr_value = df['atr'].iloc[-1]
        report = (
            f"WaveGrok Analysis for {symbol} ({primary_tf})\n"
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
            f"- Pivot High: {df['pivot_high'].iloc[-1]:.2f}, Low: {df['pivot_low'].iloc[-1]:.2f}\n"
            f"Extras:\n"
            f"- Moon Phase: {df['moon_phase'].iloc[-1]:.2f}\n\n"
            f"Trade Executed: {trade_msg}\n"
            f"Portfolio Status: {self.report_performance()}"
        )
        return report

agent = WaveGrok()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch', methods=['POST'])
def fetch_data():
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')
    limit = int(data.get('limit'))
    result = agent.fetch_data(symbol, timeframe, limit)
    if "Fetched" in result:
        agent.find_waves(timeframe)
    return jsonify({"message": result})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    symbol = data.get('symbol')
    primary_tf = data.get('primary_tf')
    secondary_tf = data.get('secondary_tf')
    result = agent.analyze_waves(symbol, primary_tf, secondary_tf)
    return jsonify({"message": result})

@app.route('/chart/<timeframe>')
def get_chart(timeframe):
    img = agent.plot_chart(timeframe)
    if img is None:
        return jsonify({"error": "No data"}), 400
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)