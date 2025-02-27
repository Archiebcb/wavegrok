from flask import Flask, request, jsonify, send_file, render_template
import ccxt
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
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
            df['momentum'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            bb = BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['adx'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
            ichimoku = IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_cloud'] = df['ichimoku_a'] - df['ichimoku_b']
            df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
            df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
            df['fractal'] = self._calculate_fractal(df['close'])
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

    def find_waves(self, timeframe, min_distance=3):
        if timeframe not in self.closes:
            return f"No data for {timeframe}—fetch some first!"
        closes = self.closes[timeframe]
        self.peaks[timeframe], _ = find_peaks(closes, distance=min_distance, prominence=closes.std()/10)
        self.troughs[timeframe], _ = find_peaks(-closes, distance=min_distance, prominence=closes.std()/10)
        return f"{timeframe}: Found {len(self.peaks[timeframe])} peaks and {len(self.troughs[timeframe])} troughs."

    def plot_chart(self, timeframe):
        df = self.data[timeframe]
        closes = self.closes[timeframe]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(closes, label='Price', color='cyan')
        ax1.plot(self.peaks[timeframe], closes[self.peaks[timeframe]], "x", label='Peaks', color='lime')
        ax1.plot(self.troughs[timeframe], closes[self.troughs[timeframe]], "o", label='Troughs', color='magenta')
        ax1.legend()
        ax1.set_title(f"WaveGrok - {timeframe}")
        ax2.plot(df['rsi'], label='RSI', color='purple')
        ax2.axhline(70, ls='--', color='red')
        ax2.axhline(30, ls='--', color='green')
        ax2.legend()
        plt.tight_layout()

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
        df = self.data[primary_tf]
        closes = self.closes[primary_tf]
        peaks = self.peaks[primary_tf]
        troughs = self.troughs[primary_tf]
        if len(peaks) < 2 or len(troughs) < 2:
            return f"Not enough peaks or troughs in {primary_tf}."

        features_rf = [df['momentum'].iloc[-1] or 0,
                       (closes[peaks[-1]] - closes[troughs[-1]]) / (closes[peaks[-1]] - closes[troughs[-2]]) if len(troughs) > 1 else 0,
                       df['volume_change'].iloc[-1] or 0, df['rsi'].iloc[-1] or 0, df['macd'].iloc[-1] or 0,
                       df['bb_width'].iloc[-1] or 0, df['atr'].iloc[-1] or 0, df['adx'].iloc[-1] or 0,
                       df['stoch_k'].iloc[-1] or 0, df['cci'].iloc[-1] or 0, df['ichimoku_cloud'].iloc[-1] or 0,
                       df['vwap_diff'].iloc[-1] or 0]
        rf_pred = self.rf_model.predict([features_rf])[0]
        lstm_input = np.array(df[['momentum', 'rsi', 'macd', 'bb_width', 'atr', 'adx', 'stoch_k', 'cci', 'ichimoku_cloud', 'vwap_diff', 'close', 'volume_change']].iloc[-10:].fillna(0)).reshape(1, 10, 12)
        lstm_pred = np.argmax(self.lstm_model.predict(lstm_input, verbose=0))
        lstm_pred = ["Wave 1", "Wave 2", "Wave 3", "Wave 4", "Wave 5", "Wave A", "Wave B", "Wave C"][lstm_pred]
        current_wave = rf_pred if random.random() > 0.3 else lstm_pred
        direction = "Up" if current_wave in ["Wave 1", "Wave 3", "Wave 5"] else "Down" if current_wave in ["Wave A", "Wave C"] else "Unclear"

        state = (current_wave, direction, df['rsi'].iloc[-1] > 70 if df['rsi'].iloc[-1] else False,
                 df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] if df['macd'].iloc[-1] else False,
                 "Above" if closes[-1] > df['bb_upper'].iloc[-1] else "Below" if closes[-1] < df['bb_lower'].iloc[-1] else "Within",
                 *self.get_meme_hype(symbol), df['moon_phase'].iloc[-1] > 0.9 if df['moon_phase'].iloc[-1] else False,
                 df['adx'].iloc[-1] > 25 if df['adx'].iloc[-1] else False, df['stoch_k'].iloc[-1] > 80 if df['stoch_k'].iloc[-1] else False,
                 df['cci'].iloc[-1] > 100 if df['cci'].iloc[-1] else False, df['ichimoku_cloud'].iloc[-1] > 0 if df['ichimoku_cloud'].iloc[-1] else False,
                 df['vwap_diff'].iloc[-1] > 0 if df['vwap_diff'].iloc[-1] else False)
        action = self.get_q_action(state)
        mtf_confirmed = secondary_tf in self.peaks and len(self.peaks[secondary_tf]) >= 2
        confidence = 0.9 if mtf_confirmed else 0.7
        trade_msg, reward = self.auto_trade(symbol, action, closes[-1], confidence)
        self.update_q_table(state, action, reward, state)
        self.epsilon = max(0.1, self.epsilon * 0.995)

        return (f"Wave: {current_wave}, Direction: {direction}\n"
                f"Action: {action}\nTrade: {trade_msg}\n"
                f"Portfolio: {self.report_performance()}")

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
    if timeframe not in agent.data:
        return jsonify({"error": "No data"}), 400
    agent.plot_chart(timeframe)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)