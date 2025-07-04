import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import re
import warnings
import ta
from scipy import stats
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden states
        h0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_1 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        h0_2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_2 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        h0_3 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0_3 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM layers
        out, _ = self.lstm1(x, (h0_1, c0_1))
        out = self.dropout(out)
        
        out, _ = self.lstm2(out, (h0_2, c0_2))
        out = self.dropout(out)
        
        out, _ = self.lstm3(out, (h0_3, c0_3))
        out = self.dropout(out)
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

class ComprehensiveShareMarketPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.models = {}
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Common exchange suffixes for different markets
        self.exchange_suffixes = {
            'Indian': ['.NS', '.BO'],  # NSE and BSE
            'US': [''],  # No suffix for US stocks
            'UK': ['.L'],  # London Stock Exchange
            'Canada': ['.TO'],  # Toronto Stock Exchange
            'Australia': ['.AX'],  # Australian Securities Exchange
        }
    
    def safe_format_number(self, value, default='N/A', format_type='comma'):
        """Safely format numbers with proper error handling"""
        if value is None or value == 'N/A' or (isinstance(value, str) and value.lower() in ['n/a', 'none', '']):
            return default
        
        try:
            if isinstance(value, str):
                # Try to convert string to number
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            
            if isinstance(value, (int, float)) and not np.isnan(value):
                if format_type == 'comma':
                    return f"{value:,.0f}"
                elif format_type == 'percentage':
                    return f"{value:.2%}"
                elif format_type == 'decimal':
                    return f"{value:.2f}"
                else:
                    return str(value)
            else:
                return default
        except (ValueError, TypeError):
            return default
    
    def safe_format_currency(self, value, currency_symbol, default='N/A'):
        """Safely format currency values"""
        if value is None or value == 'N/A':
            return default
        
        try:
            if isinstance(value, str):
                value = float(value)
            
            if isinstance(value, (int, float)) and not np.isnan(value):
                if value >= 1e12:  # Trillion
                    return f"{currency_symbol}{value/1e12:.2f}T"
                elif value >= 1e9:  # Billion
                    return f"{currency_symbol}{value/1e9:.2f}B"
                elif value >= 1e6:  # Million
                    return f"{currency_symbol}{value/1e6:.2f}M"
                elif value >= 1e3:  # Thousand
                    return f"{currency_symbol}{value/1e3:.2f}K"
                else:
                    return f"{currency_symbol}{value:.2f}"
            else:
                return default
        except (ValueError, TypeError):
            return default

    def search_company_symbol(self, company_name):
        """Advanced company symbol search with multiple strategies"""
        company_name = company_name.strip().upper()
        possible_symbols = []
        
        # Method 1: Direct search
        possible_symbols.append(company_name)
        
        # Method 2: Try with exchange suffixes
        for exchange, suffixes in self.exchange_suffixes.items():
            for suffix in suffixes:
                possible_symbols.append(f"{company_name}{suffix}")
        
        # Method 3: Abbreviated versions
        words = company_name.split()
        if len(words) > 1:
            possible_symbols.append(words[0])
            abbreviation = ''.join([word[0] for word in words])
            possible_symbols.append(abbreviation)
            
            for exchange, suffixes in self.exchange_suffixes.items():
                for suffix in suffixes:
                    possible_symbols.append(f"{words[0]}{suffix}")
                    possible_symbols.append(f"{abbreviation}{suffix}")
        
        # Method 4: Clean variations
        clean_name = re.sub(r'\b(LTD|LIMITED|INC|CORP|CORPORATION|COMPANY|CO)\b', '', company_name)
        clean_name = clean_name.strip()
        if clean_name != company_name:
            possible_symbols.append(clean_name)
            for exchange, suffixes in self.exchange_suffixes.items():
                for suffix in suffixes:
                    possible_symbols.append(f"{clean_name}{suffix}")
        
        # Test each possible symbol
        for symbol in possible_symbols:
            try:
                ticker = yf.Ticker(symbol)
                test_data = ticker.history(period="5d")
                if not test_data.empty and len(test_data) > 0:
                    info = ticker.info
                    company_long_name = info.get('longName', '').upper()
                    company_short_name = info.get('shortName', '').upper()
                    
                    if (company_name in company_long_name or 
                        company_long_name in company_name or
                        company_name in company_short_name or
                        any(word in company_long_name for word in company_name.split()) or
                        symbol.replace('.NS', '').replace('.BO', '') == company_name):
                        return symbol, info
            except Exception:
                continue
        
        return None, None
    
    def fetch_realtime_data(self, search_input, period="1y"):
        """Fetch comprehensive real-time stock data"""
        try:
            symbol, company_info = self.search_company_symbol(search_input)
            
            if symbol is None:
                symbol = search_input.upper()
                ticker = yf.Ticker(symbol)
                test_data = ticker.history(period="5d")
                if test_data.empty:
                    return None, None, f"Could not find any stock with the name or symbol: {search_input}"
                company_info = ticker.info
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None, None, f"No data available for {symbol}"
            
            self.data = data
            return data, symbol, None
            
        except Exception as e:
            return None, None, f"Error fetching data: {str(e)}"
    
    def get_stock_info(self, symbol):
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_info = self.categorize_stock(symbol, info)
            info.update(market_info)
            return info
        except Exception as e:
            st.error(f"Error fetching stock info: {e}")
            return {}
    
    def categorize_stock(self, symbol, info):
        """Categorize stock by market and exchange"""
        market_info = {
            'exchange': 'Unknown',
            'market_category': 'Unknown',
            'currency': 'USD',
            'country': 'Unknown'
        }
        
        if symbol.endswith('.NS'):
            market_info.update({
                'exchange': 'NSE (National Stock Exchange)',
                'market_category': 'Indian Market',
                'currency': 'INR',
                'country': 'India'
            })
        elif symbol.endswith('.BO'):
            market_info.update({
                'exchange': 'BSE (Bombay Stock Exchange)',
                'market_category': 'Indian Market',
                'currency': 'INR',
                'country': 'India'
            })
        elif symbol.endswith('.L'):
            market_info.update({
                'exchange': 'LSE (London Stock Exchange)',
                'market_category': 'UK Market',
                'currency': 'GBP',
                'country': 'United Kingdom'
            })
        elif symbol.endswith('.TO'):
            market_info.update({
                'exchange': 'TSX (Toronto Stock Exchange)',
                'market_category': 'Canadian Market',
                'currency': 'CAD',
                'country': 'Canada'
            })
        elif symbol.endswith('.AX'):
            market_info.update({
                'exchange': 'ASX (Australian Securities Exchange)',
                'market_category': 'Australian Market',
                'currency': 'AUD',
                'country': 'Australia'
            })
        else:
            market_info.update({
                'exchange': 'US Market (NASDAQ/NYSE)',
                'market_category': 'US Market',
                'currency': 'USD',
                'country': 'United States'
            })
        
        exchange_from_yf = info.get('exchange', '')
        if exchange_from_yf:
            market_info['exchange'] = f"{market_info['exchange']} ({exchange_from_yf})"
        
        return market_info
    
    def get_currency_symbol(self, currency):
        """Get currency symbol for display"""
        currency_symbols = {
            'USD': '$', 'INR': '₹', 'GBP': '£', 'CAD': 'C$', 
            'AUD': 'A$', 'EUR': '€', 'JPY': '¥'
        }
        return currency_symbols.get(currency, currency)
    
    def calculate_advanced_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_100'] = df['Close'].rolling(window=100).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_%R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Money Flow Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(df['Close'] > df['Close'].shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(df['Close'] < df['Close'].shift(1), 0).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + positive_flow / negative_flow))
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
        
        # Ichimoku Cloud
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        df['Ichimoku_Conversion'] = (nine_period_high + nine_period_low) / 2
        
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Ichimoku_Base'] = (period26_high + period26_low) / 2
        
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Additional features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Volume_Price_Trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
        
        return df
    
    def fibonacci_retracement(self, data):
        """Calculate Fibonacci retracement levels"""
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        
        levels = {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
        return levels
    
    def calculate_support_resistance(self, data, window=20):
        """Calculate dynamic support and resistance levels"""
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])
        
        return {
            'resistance': sorted(set(resistance_levels), reverse=True)[:5],
            'support': sorted(set(support_levels))[:5]
        }
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(data[['Close']].values)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train_advanced_models(self, data):
        """Train multiple advanced prediction models"""
        df = self.calculate_advanced_technical_indicators(data)
        df = df.dropna()
        
        # Enhanced feature set
        features = ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                   '%K', '%D', 'Williams_%R', 'ATR', 'CCI', 'Volume_Ratio', 'MFI',
                   'BB_Position', 'BB_Width', 'Price_Change', 'High_Low_Ratio']
        
        # Remove features with all NaN values
        available_features = [f for f in features if f in df.columns and not df[f].isna().all()]
        
        X = df[available_features].values
        y = df['Close'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train multiple models
        models_to_train = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR(kernel='rbf')
        }
        
        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            self.models[name] = model
        
        # Train LSTM
        X_lstm, y_lstm = self.prepare_lstm_data(df)
        
        if len(X_lstm) > 100:
            lstm_train_size = int(len(X_lstm) * 0.8)
            X_lstm_train = X_lstm[:lstm_train_size]
            y_lstm_train = y_lstm[:lstm_train_size]
            
            with st.spinner("Training LSTM model..."):
                lstm_model = self.train_lstm_model(X_lstm_train, y_lstm_train)
            self.models['lstm'] = lstm_model
        
        return df
    
    def train_lstm_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train PyTorch LSTM model"""
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = LSTMModel().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return model
    
    def predict_with_ensemble(self, data, model_types=['random_forest', 'gradient_boosting']):
        """Ensemble prediction using multiple models"""
        df = self.calculate_advanced_technical_indicators(data)
        df = df.dropna()
        
        predictions = []
        
        for model_type in model_types:
            if model_type in self.models:
                if model_type == 'lstm':
                    scaled_data = self.scaler.transform(df[['Close']].values)
                    if len(scaled_data) >= 60:
                        last_60_days = scaled_data[-60:]
                        X_tensor = torch.FloatTensor(last_60_days).unsqueeze(0).unsqueeze(-1).to(self.device)
                        
                        self.models['lstm'].eval()
                        with torch.no_grad():
                            pred_scaled = self.models['lstm'](X_tensor).cpu().numpy()
                        
                        pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                        predictions.append(pred)
                else:
                    features = ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                               '%K', '%D', 'Williams_%R', 'ATR', 'CCI', 'Volume_Ratio', 'MFI',
                               'BB_Position', 'BB_Width', 'Price_Change', 'High_Low_Ratio']
                    
                    available_features = [f for f in features if f in df.columns and not df[f].isna().all()]
                    last_features = df[available_features].iloc[-1:].values
                    pred = self.models[model_type].predict(last_features)[0]
                    predictions.append(pred)
        
        return np.mean(predictions) if predictions else data['Close'][-1]
    
    def calculate_comprehensive_signals(self, data):
        """Generate comprehensive trading signals"""
        df = self.calculate_advanced_technical_indicators(data)
        signals = []
        
        for i in range(len(df)):
            signal_score = 0
            signal_reasons = []
            
            # RSI signals
            if not pd.isna(df['RSI'].iloc[i]):
                if df['RSI'].iloc[i] < 30:
                    signal_score += 2
                    signal_reasons.append("RSI Oversold")
                elif df['RSI'].iloc[i] > 70:
                    signal_score -= 2
                    signal_reasons.append("RSI Overbought")
            
            # MACD signals
            if i > 0 and not pd.isna(df['MACD'].iloc[i]):
                if (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and 
                    df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]):
                    signal_score += 1
                    signal_reasons.append("MACD Bullish Cross")
                elif (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and 
                      df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]):
                    signal_score -= 1
                    signal_reasons.append("MACD Bearish Cross")
            
            # Moving Average signals
            if i > 0 and not pd.isna(df['MA_20'].iloc[i]):
                if df['Close'].iloc[i] > df['MA_20'].iloc[i] and df['Close'].iloc[i-1] <= df['MA_20'].iloc[i-1]:
                    signal_score += 1
                    signal_reasons.append("Price Above MA20")
                elif df['Close'].iloc[i] < df['MA_20'].iloc[i] and df['Close'].iloc[i-1] >= df['MA_20'].iloc[i-1]:
                    signal_score -= 1
                    signal_reasons.append("Price Below MA20")
            
            # Stochastic signals
            if not pd.isna(df['%K'].iloc[i]):
                if df['%K'].iloc[i] < 20:
                    signal_score += 1
                    signal_reasons.append("Stochastic Oversold")
                elif df['%K'].iloc[i] > 80:
                    signal_score -= 1
                    signal_reasons.append("Stochastic Overbought")
            
            # Bollinger Bands signals
            if not pd.isna(df['BB_Position'].iloc[i]):
                if df['BB_Position'].iloc[i] < 0.1:
                    signal_score += 1
                    signal_reasons.append("BB Lower Band Touch")
                elif df['BB_Position'].iloc[i] > 0.9:
                    signal_score -= 1
                    signal_reasons.append("BB Upper Band Touch")
            
            # Determine final signal
            if signal_score >= 3:
                signal = "Strong Buy"
            elif signal_score >= 1:
                signal = "Buy"
            elif signal_score <= -3:
                signal = "Strong Sell"
            elif signal_score <= -1:
                signal = "Sell"
            else:
                signal = "Hold"
            
            signals.append({
                'signal': signal,
                'score': signal_score,
                'reasons': ', '.join(signal_reasons) if signal_reasons else 'No clear signals'
            })
        
        df['Signal'] = [s['signal'] for s in signals]
        df['Signal_Score'] = [s['score'] for s in signals]
        df['Signal_Reasons'] = [s['reasons'] for s in signals]
        
        return df
    
    def advanced_risk_analysis(self, data):
        """Comprehensive risk analysis"""
        returns = data['Close'].pct_change().dropna()
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        max_drawdown = ((data['Close'] / data['Close'].cummax()) - 1).min()
        
        # Advanced risk metrics
        sortino_ratio = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252))
        calmar_ratio = (returns.mean() * 252) / abs(max_drawdown)
        
        # VaR calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Beta calculation (if possible)
        try:
            # Using SPY as market benchmark
            spy = yf.download("SPY", period="1y")['Close']
            market_returns = spy.pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            stock_returns_aligned = returns.loc[common_dates]
            market_returns_aligned = market_returns.loc[common_dates]
            
            beta = np.cov(stock_returns_aligned, market_returns_aligned)[0][1] / np.var(market_returns_aligned)
        except:
            beta = None
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'beta': beta
        }
    
    def model_performance_analysis(self, data):
        """Comprehensive model performance comparison"""
        df = self.calculate_advanced_technical_indicators(data)
        df = df.dropna()
        
        features = ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
                   '%K', '%D', 'Williams_%R', 'ATR', 'CCI', 'Volume_Ratio', 'MFI',
                   'BB_Position', 'BB_Width', 'Price_Change', 'High_Low_Ratio']
        
        available_features = [f for f in features if f in df.columns and not df[f].isna().all()]
        X = df[available_features].values
        y = df['Close'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        performance = {}
        
        for name, model in self.models.items():
            if name != 'lstm':
                y_pred = model.predict(X_test)
                performance[name] = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
                }
        
        return performance

def create_comprehensive_dashboard():
    """Create comprehensive Streamlit dashboard"""
    st.set_page_config(
        page_title="Complete Share Market Analysis Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .help-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Complete Share Market Analysis Dashboard</h1>
        <p>Your One-Stop Solution for Stock Analysis, Prediction & Trading Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = ComprehensiveShareMarketPredictor()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Device info
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"Computing Device: {device_info}")
    
    # Main search
    st.subheader("🔍 Company Search")
    
    # Search help section
    with st.expander("📚 Company Search Help", expanded=False):
        st.markdown("""
        <div class="help-section">
        <h4>🔍 How to Search for Companies</h4>
        
        <h5>What:</h5>
        <p>Search for any publicly traded company worldwide by name or stock symbol.</p>
        
        <h5>Why:</h5>
        <p>This intelligent search system finds companies across global exchanges automatically, 
        eliminating the need to know exact symbols or exchange suffixes.</p>
        
        <h5>How to Use:</h5>
        <ul>
            <li><b>Company Name:</b> Type full name (e.g., "Apple Inc", "Reliance Industries")</li>
            <li><b>Stock Symbol:</b> Use direct symbols (e.g., "AAPL", "MSFT")</li>
            <li><b>Partial Names:</b> Try abbreviations (e.g., "Apple" for Apple Inc)</li>
            <li><b>Global Markets:</b> Works for US, Indian, UK, Canadian, Australian markets</li>
        </ul>
        
        <h5>Examples:</h5>
        <ul>
            <li>🇺🇸 US: "Apple", "AAPL", "Microsoft", "MSFT"</li>
            <li>🇮🇳 India: "Reliance", "RELIANCE.NS", "TCS", "TCS.BO"</li>
            <li>🇬🇧 UK: "BP.L", "Vodafone"</li>
            <li>🇨🇦 Canada: "Shopify.TO"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    search_input = st.text_input(
        "Enter Company Name or Stock Symbol", 
        placeholder="e.g., Apple, Microsoft, Reliance, TCS, AAPL, MSFT, GOOGL...",
        help="Search any company worldwide by name or symbol"
    )
    
    # Configuration
    config_col1, config_col2 = st.columns([1, 1])
    with config_col1:
        period = st.selectbox("📅 Data Period", ["1y", "6mo", "3mo", "1mo", "2y", "5y"])
    with config_col2:
        analysis_type = st.selectbox("📊 Analysis Type", 
                                   ["Complete Analysis", "Quick Overview", "Technical Only", "Fundamental Only"])
    
    # Configuration help
    with st.expander("⚙️ Configuration Help", expanded=False):
        st.markdown("""
        <div class="help-section">
        <h4>📅 Data Period</h4>
        <p><b>What:</b> The historical time range for analysis</p>
        <p><b>Why:</b> Different periods reveal different patterns - short-term for day trading, long-term for investing</p>
        <ul>
            <li><b>1mo:</b> Short-term trading analysis</li>
            <li><b>3mo:</b> Quarterly trend analysis</li>
            <li><b>6mo:</b> Medium-term patterns</li>
            <li><b>1y:</b> Annual trends (recommended)</li>
            <li><b>2y-5y:</b> Long-term investment analysis</li>
        </ul>
        
        <h4>📊 Analysis Type</h4>
        <ul>
            <li><b>Complete Analysis:</b> All features - technical, fundamental, predictions</li>
            <li><b>Quick Overview:</b> Essential metrics only</li>
            <li><b>Technical Only:</b> Charts, indicators, signals</li>
            <li><b>Fundamental Only:</b> Company financials and valuation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced options
    st.write("⚙️ Advanced Options")
    
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            show_fibonacci = st.checkbox("Fibonacci Retracement", value=True)
        with col2:
            show_support_resistance = st.checkbox("Support/Resistance", value=True)
        with col3:
            ensemble_prediction = st.checkbox("Ensemble Prediction", value=True)
    
    # Advanced options help
    with st.expander("🔧 Advanced Features Help", expanded=False):
        st.markdown("""
        <div class="help-section">
        <h4>📐 Fibonacci Retracement</h4>
        <p><b>What:</b> Mathematical levels based on Fibonacci sequence (23.6%, 38.2%, 50%, 61.8%)</p>
        <p><b>Why:</b> Markets often reverse at these levels due to psychological factors</p>
        <p><b>How to Interpret:</b> Price bounces or breaks through these levels indicate support/resistance</p>
        
        <h4>🎯 Support & Resistance</h4>
        <p><b>What:</b> Price levels where stock historically bounces up (support) or down (resistance)</p>
        <p><b>Why:</b> These levels represent psychological barriers for traders</p>
        <p><b>How to Use:</b> Buy near support, sell near resistance, breakouts signal trend changes</p>
        
        <h4>🤖 Ensemble Prediction</h4>
        <p><b>What:</b> Combines multiple AI models for better accuracy</p>
        <p><b>Why:</b> Reduces individual model bias and improves prediction reliability</p>
        <p><b>How it Works:</b> Averages predictions from Random Forest, Gradient Boosting, and LSTM models</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Example searches
    st.markdown("### 💡 Example Searches")
    examples_col1, examples_col2, examples_col3, examples_col4 = st.columns(4)
    
    with examples_col1:
        if st.button("🍎 Apple Inc"):
            search_input = "AAPL"
    with examples_col2:
        if st.button("🏭 Reliance Industries"):
            search_input = "RELIANCE.NS"
    with examples_col3:
        if st.button("💻 Microsoft"):
            search_input = "MSFT"
    with examples_col4:
        if st.button("🏦 HDFC Bank"):
            search_input = "HDFCBANK.NS"
    
    # Main analysis
    if st.button("🚀 Start Analysis", type="primary") and search_input:
        with st.spinner(f"🔄 Analyzing '{search_input}'..."):
            data, symbol, error = predictor.fetch_realtime_data(search_input, period)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 **Suggestions:**")
                st.write("• Try the exact stock symbol (e.g., 'AAPL' for Apple)")
                st.write("• Search with full company name")
                st.write("• Check spelling and try variations")
            
            elif data is not None and len(data) > 0:
                # Get comprehensive stock info
                stock_info = predictor.get_stock_info(symbol)
                currency_symbol = predictor.get_currency_symbol(stock_info.get('currency', 'USD'))
                company_name = stock_info.get('longName', stock_info.get('shortName', search_input))
                
                # Company header
                st.markdown(f"## 📊 {company_name}")
                st.markdown(f"**Symbol:** `{symbol}` | **Exchange:** {stock_info.get('exchange', 'Unknown')}")
                
                # Key metrics
                st.markdown("### 💰 Key Metrics")
                
                # Key metrics help
                with st.expander("💰 Key Metrics Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>📊 Understanding Key Metrics</h4>
                    
                    <h5>💰 Current Price</h5>
                    <p><b>What:</b> Latest trading price of the stock</p>
                    <p><b>Why Important:</b> Shows current market valuation</p>
                    
                    <h5>📈 Daily Change</h5>
                    <p><b>What:</b> Price difference from previous day's close</p>
                    <p><b>How to Interpret:</b></p>
                    <ul>
                        <li>🟢 Green: Stock is up (positive sentiment)</li>
                        <li>🔴 Red: Stock is down (negative sentiment)</li>
                        <li>Percentage shows magnitude of change</li>
                    </ul>
                    
                    <h5>📊 Volume</h5>
                    <p><b>What:</b> Number of shares traded today</p>
                    <p><b>Why Important:</b> High volume confirms price moves, low volume suggests weak trends</p>
                    
                    <h5>🔝 52W High/Low</h5>
                    <p><b>What:</b> Highest and lowest prices in past 52 weeks</p>
                    <p><b>How to Use:</b> Compare current price to these levels for context</p>
                    
                    <h5>💹 Market Cap</h5>
                    <p><b>What:</b> Total value of all company shares (Price × Outstanding Shares)</p>
                    <p><b>Company Size:</b></p>
                    <ul>
                        <li>Large Cap: >$10B (Stable, established companies)</li>
                        <li>Mid Cap: $2B-$10B (Growth potential with some risk)</li>
                        <li>Small Cap: <$2B (High growth potential, higher risk)</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Key metrics row
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                current_price = data['Close'][-1]
                previous_close = data['Close'][-2]
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
                
                with col1:
                    st.metric("💰 Current Price", f"{currency_symbol}{current_price:.2f}")
                with col2:
                    st.metric("📈 Daily Change", f"{currency_symbol}{change:.2f}", f"{change_percent:+.2f}%")
                with col3:
                    st.metric("📊 Volume", predictor.safe_format_number(data['Volume'][-1]))
                with col4:
                    fifty_two_high = predictor.safe_format_currency(stock_info.get('fiftyTwoWeekHigh'), currency_symbol)
                    st.metric("🔝 52W High", fifty_two_high)
                with col5:
                    fifty_two_low = predictor.safe_format_currency(stock_info.get('fiftyTwoWeekLow'), currency_symbol)
                    st.metric("🔻 52W Low", fifty_two_low)
                with col6:
                    market_cap = predictor.safe_format_currency(stock_info.get('marketCap'), currency_symbol)
                    st.metric("💹 Market Cap", market_cap)
                
                # Train models
                df_indicators = predictor.train_advanced_models(data)
                
                # Predictions section
                st.markdown("## 🔮 Price Predictions")
                
                # Predictions help
                with st.expander("🔮 Price Predictions Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>🤖 AI-Powered Price Predictions</h4>
                    
                    <h5>What Are These Predictions?</h5>
                    <p>Our system uses advanced machine learning models to forecast next-day stock prices:</p>
                    <ul>
                        <li><b>Random Forest:</b> Uses multiple decision trees for robust predictions</li>
                        <li><b>Gradient Boosting:</b> Learns from prediction errors progressively</li>
                        <li><b>LSTM Neural Network:</b> Deep learning model that remembers long-term patterns</li>
                        <li><b>Ensemble:</b> Combines all models for best accuracy</li>
                    </ul>
                    
                    <h5>How to Interpret:</h5>
                    <ul>
                        <li><b>Predicted Price:</b> Expected closing price for next trading day</li>
                        <li><b>Change Amount:</b> Expected price movement in currency units</li>
                        <li><b>Change Percentage:</b> Expected movement as percentage</li>
                        <li><b>Confidence Level:</b> Model's certainty (60-95%)</li>
                    </ul>
                    
                    <h5>Important Notes:</h5>
                    <ul>
                        <li>⚠️ Predictions are based on historical patterns</li>
                        <li>📰 News and events can cause sudden changes</li>
                        <li>🎯 Use as guidance, not absolute truth</li>
                        <li>💡 Combine with technical and fundamental analysis</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    if ensemble_prediction:
                        next_day_pred = predictor.predict_with_ensemble(data)
                        st.success(f"🎯 **Ensemble Prediction**\n{currency_symbol}{next_day_pred:.2f}")
                    else:
                        next_day_pred = predictor.predict_with_ensemble(data, ['random_forest'])
                        st.info(f"🤖 **ML Prediction**\n{currency_symbol}{next_day_pred:.2f}")
                
                with pred_col2:
                    pred_change = next_day_pred - current_price
                    pred_change_percent = (pred_change / current_price) * 100
                    
                    color = "🟢" if pred_change > 0 else "🔴" if pred_change < 0 else "⚪"
                    st.info(f"{color} **Predicted Change**\n{currency_symbol}{pred_change:+.2f} ({pred_change_percent:+.2f}%)")
                
                with pred_col3:
                    confidence = min(95, max(60, 80 + abs(pred_change_percent) * 2))
                    st.info(f"📊 **Confidence Level**\n{confidence:.1f}%")
                
                # Technical Analysis
                if analysis_type in ["Complete Analysis", "Technical Only"]:
                    st.markdown("## 📈 Technical Analysis")
                    
                    # Technical analysis help
                    with st.expander("📈 Technical Analysis Help", expanded=False):
                        st.markdown("""
                        <div class="help-section">
                        <h4>📊 Understanding Technical Analysis</h4>
                        
                        <h5>🕯️ Candlestick Chart</h5>
                        <p><b>What:</b> Shows Open, High, Low, Close prices for each time period</p>
                        <p><b>How to Read:</b></p>
                        <ul>
                            <li>🟢 Green Candle: Close > Open (bullish)</li>
                            <li>🔴 Red Candle: Close < Open (bearish)</li>
                            <li>Wick lengths show price volatility</li>
                        </ul>
                        
                        <h5>📈 Moving Averages (MA)</h5>
                        <p><b>What:</b> Average price over specific periods (20, 50 days)</p>
                        <p><b>How to Use:</b></p>
                        <ul>
                            <li>Price above MA = Uptrend</li>
                            <li>Price below MA = Downtrend</li>
                            <li>MA crossovers signal trend changes</li>
                        </ul>
                        
                        <h5>🎯 Bollinger Bands</h5>
                        <p><b>What:</b> Price channels based on standard deviation</p>
                        <p><b>Interpretation:</b></p>
                        <ul>
                            <li>Price near upper band = Potentially overbought</li>
                            <li>Price near lower band = Potentially oversold</li>
                            <li>Band squeeze = Low volatility, potential breakout</li>
                        </ul>
                        
                        <h5>📊 RSI (Relative Strength Index)</h5>
                        <p><b>What:</b> Momentum oscillator (0-100)</p>
                        <p><b>Interpretation:</b></p>
                        <ul>
                            <li>RSI > 70 = Overbought (potential sell signal)</li>
                            <li>RSI < 30 = Oversold (potential buy signal)</li>
                            <li>RSI 30-70 = Neutral zone</li>
                        </ul>
                        
                        <h5>📈 MACD</h5>
                        <p><b>What:</b> Moving Average Convergence Divergence</p>
                        <p><b>Signals:</b></p>
                        <ul>
                            <li>MACD line above signal line = Bullish</li>
                            <li>MACD line below signal line = Bearish</li>
                            <li>Crossovers indicate potential trend changes</li>
                        </ul>
                        
                        <h5>📊 Volume</h5>
                        <p><b>What:</b> Number of shares traded</p>
                        <p><b>Why Important:</b></p>
                        <ul>
                            <li>High volume confirms price moves</li>
                            <li>Low volume suggests weak trends</li>
                            <li>Volume spikes often precede major moves</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Main price chart with all indicators
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Volume'),
                        row_width=[0.2, 0.1, 0.1, 0.1]
                    )
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Price"
                    ), row=1, col=1)
                    
                    # Moving averages
                    if 'MA_20' in df_indicators.columns:
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MA_20'], 
                                               name="MA20", line=dict(color='orange')), row=1, col=1)
                    if 'MA_50' in df_indicators.columns:
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MA_50'], 
                                               name="MA50", line=dict(color='blue')), row=1, col=1)
                    
                    # Bollinger Bands
                    if all(col in df_indicators.columns for col in ['BB_Upper', 'BB_Lower']):
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['BB_Upper'], 
                                               name="BB Upper", line=dict(color='red', dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['BB_Lower'], 
                                               name="BB Lower", line=dict(color='red', dash='dash')), row=1, col=1)
                    
                    # Support and Resistance
                    if show_support_resistance:
                        sr_levels = predictor.calculate_support_resistance(data)
                        for resistance in sr_levels['resistance'][:3]:
                            fig.add_hline(y=resistance, line_dash="dash", line_color="red", 
                                        annotation_text=f"R: {resistance:.2f}", row=1, col=1)
                        for support in sr_levels['support'][:3]:
                            fig.add_hline(y=support, line_dash="dash", line_color="green", 
                                        annotation_text=f"S: {support:.2f}", row=1, col=1)
                    
                    # Fibonacci levels
                    if show_fibonacci:
                        fib_levels = predictor.fibonacci_retracement(data)
                        for level, price in fib_levels.items():
                            fig.add_hline(y=price, line_dash="dot", line_color="purple", 
                                        annotation_text=f"Fib {level}: {price:.2f}", row=1, col=1)
                    
                    # RSI
                    if 'RSI' in df_indicators.columns:
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['RSI'], 
                                               name="RSI", line=dict(color='purple')), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # MACD
                    if all(col in df_indicators.columns for col in ['MACD', 'MACD_Signal']):
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD'], 
                                               name="MACD", line=dict(color='blue')), row=3, col=1)
                        fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators['MACD_Signal'], 
                                               name="Signal", line=dict(color='red')), row=3, col=1)
                    
                    # Volume
                    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", 
                                       marker_color='lightblue'), row=4, col=1)
                    
                    fig.update_layout(height=1000, title=f"{company_name} - Complete Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical indicators summary
                    tech_col1, tech_col2, tech_col3 = st.columns(3)
                    
                    with tech_col1:
                        st.markdown("### 📊 Current Indicators")
                        if 'RSI' in df_indicators.columns:
                            rsi_current = df_indicators['RSI'].iloc[-1]
                            rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
                            st.write(f"**RSI:** {rsi_current:.1f} ({rsi_status})")
                        
                        if 'MACD' in df_indicators.columns:
                            macd_current = df_indicators['MACD'].iloc[-1]
                            macd_signal = df_indicators['MACD_Signal'].iloc[-1]
                            macd_status = "Bullish" if macd_current > macd_signal else "Bearish"
                            st.write(f"**MACD:** {macd_status}")
                    
                    with tech_col2:
                        st.markdown("### 🎯 Support & Resistance")
                        if show_support_resistance:
                            sr_levels = predictor.calculate_support_resistance(data)
                            st.write("**Resistance Levels:**")
                            for i, level in enumerate(sr_levels['resistance'][:3]):
                                st.write(f"R{i+1}: {currency_symbol}{level:.2f}")
                            st.write("**Support Levels:**")
                            for i, level in enumerate(sr_levels['support'][:3]):
                                st.write(f"S{i+1}: {currency_symbol}{level:.2f}")
                    
                    with tech_col3:
                        st.markdown("### 📐 Fibonacci Levels")
                        if show_fibonacci:
                            fib_levels = predictor.fibonacci_retracement(data)
                            for level, price in fib_levels.items():
                                st.write(f"**{level}:** {currency_symbol}{price:.2f}")
                
                # Trading Signals
                st.markdown("## 🎯 Trading Signals")
                
                # Trading signals help
                with st.expander("🎯 Trading Signals Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>🎯 Understanding Trading Signals</h4>
                    
                    <h5>What Are Trading Signals?</h5>
                    <p>Algorithmic recommendations based on multiple technical indicators combined together.</p>
                    
                    <h5>Signal Types:</h5>
                    <ul>
                        <li>🟢 <b>Strong Buy:</b> Multiple bullish indicators align (Score: +3 or higher)</li>
                        <li>🟡 <b>Buy:</b> Some bullish indicators present (Score: +1 to +2)</li>
                        <li>⚪ <b>Hold:</b> Mixed or neutral signals (Score: 0)</li>
                        <li>🟠 <b>Sell:</b> Some bearish indicators present (Score: -1 to -2)</li>
                        <li>🔴 <b>Strong Sell:</b> Multiple bearish indicators align (Score: -3 or lower)</li>
                    </ul>
                    
                    <h5>How Signals Are Generated:</h5>
                    <ul>
                        <li><b>RSI Analysis:</b> Overbought/oversold conditions</li>
                        <li><b>MACD Crossovers:</b> Trend direction changes</li>
                        <li><b>Moving Average Position:</b> Price vs trend lines</li>
                        <li><b>Stochastic Levels:</b> Momentum indicators</li>
                        <li><b>Bollinger Band Position:</b> Volatility-based signals</li>
                    </ul>
                    
                    <h5>How to Use:</h5>
                    <ul>
                        <li>💡 Use signals as <b>guidance</b>, not absolute rules</li>
                        <li>🔍 Always check the reasons behind each signal</li>
                        <li>📊 Combine with fundamental analysis</li>
                        <li>⚠️ Consider market conditions and news</li>
                        <li>💰 Never invest more than you can afford to lose</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                signals_df = predictor.calculate_comprehensive_signals(data)
                latest_signal = signals_df.iloc[-1]
                
                signal_col1, signal_col2, signal_col3 = st.columns(3)
                
                with signal_col1:
                    signal_color = {
                        "Strong Buy": "🟢",
                        "Buy": "🟡", 
                        "Hold": "⚪",
                        "Sell": "🟠",
                        "Strong Sell": "🔴"
                    }
                    st.markdown(f"### {signal_color.get(latest_signal['Signal'], '⚪')} Current Signal")
                    st.markdown(f"**{latest_signal['Signal']}**")
                    st.write(f"Score: {latest_signal['Signal_Score']}")
                
                with signal_col2:
                    st.markdown("### 📋 Signal Reasons")
                    st.write(latest_signal['Signal_Reasons'])
                
                with signal_col3:
                    st.markdown("### 📊 Recent Signals")
                    recent_signals = signals_df[['Signal', 'Signal_Score']].tail(5)
                    st.dataframe(recent_signals)
                
                # Risk Analysis
                st.markdown("## ⚠️ Risk Analysis")
                
                # Risk analysis help
                with st.expander("⚠️ Risk Analysis Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>⚠️ Understanding Risk Metrics</h4>
                    
                    <h5>📊 Volatility</h5>
                    <p><b>What:</b> Measure of price fluctuation (annualized standard deviation)</p>
                    <p><b>Interpretation:</b></p>
                    <ul>
                        <li>Low (0-15%): Stable, conservative stocks</li>
                        <li>Medium (15-25%): Moderate risk stocks</li>
                        <li>High (25%+): Volatile, risky stocks</li>
                    </ul>
                    
                    <h5>📈 Sharpe Ratio</h5>
                    <p><b>What:</b> Risk-adjusted return measure</p>
                    <p><b>Interpretation:</b></p>
                    <ul>
                        <li>>1.0: Excellent risk-adjusted returns</li>
                        <li>0.5-1.0: Good risk-adjusted returns</li>
                        <li><0.5: Poor risk-adjusted returns</li>
                    </ul>
                    
                    <h5>📉 Maximum Drawdown</h5>
                    <p><b>What:</b> Largest peak-to-trough decline</p>
                    <p><b>Why Important:</b> Shows worst-case scenario loss</p>
                    
                    <h5>🎯 Sortino Ratio</h5>
                    <p><b>What:</b> Like Sharpe ratio but only considers downside volatility</p>
                    <p><b>Better than Sharpe:</b> Doesn't penalize upside volatility</p>
                    
                    <h5>🔻 Value at Risk (VaR)</h5>
                    <p><b>What:</b> Maximum expected loss at given confidence level</p>
                    <ul>
                        <li><b>VaR 95%:</b> 95% chance loss won't exceed this amount</li>
                        <li><b>VaR 99%:</b> 99% chance loss won't exceed this amount</li>
                    </ul>
                    
                    <h5>📊 Beta</h5>
                    <p><b>What:</b> Correlation with overall market (vs S&P 500)</p>
                    <ul>
                        <li>Beta = 1: Moves with market</li>
                        <li>Beta > 1: More volatile than market</li>
                        <li>Beta < 1: Less volatile than market</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                risk_metrics = predictor.advanced_risk_analysis(data)
                
                risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                
                with risk_col1:
                    st.metric("📊 Volatility", f"{risk_metrics['volatility']:.2%}")
                    st.metric("📈 Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
                
                with risk_col2:
                    st.metric("📉 Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                    st.metric("🎯 Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
                
                with risk_col3:
                    st.metric("🔻 VaR (95%)", f"{risk_metrics['var_95']:.2%}")
                    st.metric("🔻 VaR (99%)", f"{risk_metrics['var_99']:.2%}")
                
                with risk_col4:
                    st.metric("💹 Calmar Ratio", f"{risk_metrics['calmar_ratio']:.2f}")
                    if risk_metrics['beta']:
                        st.metric("📊 Beta", f"{risk_metrics['beta']:.2f}")
                
                # Fundamental Analysis
                if analysis_type in ["Complete Analysis", "Fundamental Only"]:
                    st.markdown("## 🏢 Fundamental Analysis")
                    
                    # Fundamental analysis help
                    with st.expander("🏢 Fundamental Analysis Help", expanded=False):
                        st.markdown("""
                        <div class="help-section">
                        <h4>🏢 Understanding Fundamental Analysis</h4>
                        
                        <h5>📊 Valuation Metrics</h5>
                        
                        <h6>P/E Ratio (Price-to-Earnings)</h6>
                        <p><b>What:</b> Stock price divided by earnings per share</p>
                        <p><b>Interpretation:</b></p>
                        <ul>
                            <li>Low P/E (5-15): Potentially undervalued or slow growth</li>
                            <li>Medium P/E (15-25): Fair valuation</li>
                            <li>High P/E (25+): Potentially overvalued or high growth expected</li>
                        </ul>
                        
                        <h6>P/B Ratio (Price-to-Book)</h6>
                        <p><b>What:</b> Market value vs book value of assets</p>
                        <ul>
                            <li>P/B < 1: Trading below book value (potential value)</li>
                            <li>P/B > 1: Trading above book value (growth premium)</li>
                        </ul>
                        
                        <h6>PEG Ratio (P/E to Growth)</h6>
                        <p><b>What:</b> P/E ratio divided by earnings growth rate</p>
                        <ul>
                            <li>PEG < 1: Potentially undervalued relative to growth</li>
                            <li>PEG > 1: Potentially overvalued relative to growth</li>
                        </ul>
                        
                        <h5>💰 Financial Health Metrics</h5>
                        
                        <h6>ROE (Return on Equity)</h6>
                        <p><b>What:</b> How efficiently company uses shareholder equity</p>
                        <ul>
                            <li>15%+: Excellent</li>
                            <li>10-15%: Good</li>
                            <li><10%: Poor</li>
                        </ul>
                        
                        <h6>ROA (Return on Assets)</h6>
                        <p><b>What:</b> How efficiently company uses its assets</p>
                        
                        <h6>Debt-to-Equity</h6>
                        <p><b>What:</b> Company's debt relative to equity</p>
                        <ul>
                            <li>Low (<0.3): Conservative, low risk</li>
                            <li>Medium (0.3-0.6): Balanced</li>
                            <li>High (>0.6): Leveraged, higher risk</li>
                        </ul>
                        
                        <h6>Current Ratio</h6>
                        <p><b>What:</b> Current assets / Current liabilities</p>
                        <ul>
                            <li>>2: Very liquid, can pay short-term debts</li>
                            <li>1-2: Adequate liquidity</li>
                            <li><1: Potential liquidity issues</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    fund_col1, fund_col2 = st.columns(2)
                    
                    with fund_col1:
                        st.markdown("### 📊 Valuation Metrics")
                        st.write(f"**P/E Ratio:** {predictor.safe_format_number(stock_info.get('trailingPE'), format_type='decimal')}")
                        st.write(f"**P/B Ratio:** {predictor.safe_format_number(stock_info.get('priceToBook'), format_type='decimal')}")
                        st.write(f"**PEG Ratio:** {predictor.safe_format_number(stock_info.get('pegRatio'), format_type='decimal')}")
                        st.write(f"**Price/Sales:** {predictor.safe_format_number(stock_info.get('priceToSalesTrailing12Months'), format_type='decimal')}")
                        st.write(f"**EV/EBITDA:** {predictor.safe_format_number(stock_info.get('enterpriseToEbitda'), format_type='decimal')}")
                    
                    with fund_col2:
                        st.markdown("### 💰 Financial Health")
                        st.write(f"**ROE:** {predictor.safe_format_number(stock_info.get('returnOnEquity'), format_type='percentage')}")
                        st.write(f"**ROA:** {predictor.safe_format_number(stock_info.get('returnOnAssets'), format_type='percentage')}")
                        st.write(f"**Debt/Equity:** {predictor.safe_format_number(stock_info.get('debtToEquity'), format_type='decimal')}")
                        st.write(f"**Current Ratio:** {predictor.safe_format_number(stock_info.get('currentRatio'), format_type='decimal')}")
                        st.write(f"**Quick Ratio:** {predictor.safe_format_number(stock_info.get('quickRatio'), format_type='decimal')}")

                # Model Performance
                st.markdown("## 🤖 Model Performance")
                
                # Model performance help
                with st.expander("🤖 Model Performance Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>🤖 Understanding Model Performance</h4>
                    
                    <h5>What Are These Metrics?</h5>
                    <p>These metrics show how well our AI models predict stock prices on historical data.</p>
                    
                    <h5>📊 Performance Metrics Explained:</h5>
                    
                    <h6>MSE (Mean Squared Error)</h6>
                    <p><b>What:</b> Average of squared prediction errors</p>
                    <p><b>Interpretation:</b> Lower is better (closer to 0 = perfect predictions)</p>
                    
                    <h6>MAE (Mean Absolute Error)</h6>
                    <p><b>What:</b> Average absolute difference between predicted and actual prices</p>
                    <p><b>Interpretation:</b> Lower is better, expressed in currency units</p>
                    
                    <h6>R² (R-Squared)</h6>
                    <p><b>What:</b> Percentage of price variation explained by the model</p>
                    <p><b>Interpretation:</b></p>
                    <ul>
                        <li>1.0 = Perfect predictions (100% explained)</li>
                        <li>0.8+ = Very good model</li>
                        <li>0.6-0.8 = Good model</li>
                        <li>0.4-0.6 = Fair model</li>
                        <li><0.4 = Poor model</li>
                    </ul>
                    
                    <h6>RMSE (Root Mean Squared Error)</h6>
                    <p><b>What:</b> Square root of MSE, in same units as price</p>
                    <p><b>Interpretation:</b> Average prediction error in currency units</p>
                    
                    <h5>🎯 Which Model to Trust?</h5>
                    <ul>
                        <li><b>Highest R²:</b> Best at explaining price patterns</li>
                        <li><b>Lowest RMSE/MAE:</b> Most accurate predictions</li>
                        <li><b>Ensemble:</b> Usually most reliable (combines all models)</li>
                    </ul>
                    
                    <h5>⚠️ Important Notes:</h5>
                    <ul>
                        <li>Past performance doesn't guarantee future results</li>
                        <li>Models work best in stable market conditions</li>
                        <li>Sudden news/events can make predictions less accurate</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                performance = predictor.model_performance_analysis(data)
                
                perf_df = pd.DataFrame(performance).T
                st.dataframe(perf_df.round(4))
                
                # Company Information
                st.markdown("## ℹ️ Company Information")
                
                # Company information help
                with st.expander("ℹ️ Company Information Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>ℹ️ Understanding Company Information</h4>
                    
                    <h5>🏢 Basic Company Data</h5>
                    <p><b>Company Name:</b> Official registered name</p>
                    <p><b>Sector:</b> Broad industry category (Technology, Healthcare, etc.)</p>
                    <p><b>Industry:</b> Specific business focus within sector</p>
                    <p><b>Country:</b> Primary country of operations</p>
                    <p><b>Exchange:</b> Stock exchange where shares are traded</p>
                    
                    <h5>👥 Employee Count</h5>
                    <p><b>What it Shows:</b> Company size and scale of operations</p>
                    <ul>
                        <li>Large (50,000+): Established multinational corporations</li>
                        <li>Medium (1,000-50,000): Growing companies with significant operations</li>
                        <li>Small (<1,000): Startups or niche businesses</li>
                    </ul>
                    
                    <h5>📞 Contact Information</h5>
                    <p>Official company contact details for investor relations and corporate information.</p>
                    
                    <h5>📝 Business Summary</h5>
                    <p>Official description of company's business model, products, services, and market position.</p>
                    
                    <h5>💡 Why This Matters</h5>
                    <ul>
                        <li><b>Investment Context:</b> Understand what you're investing in</li>
                        <li><b>Risk Assessment:</b> Industry and geography affect risk levels</li>
                        <li><b>Growth Potential:</b> Sector trends impact future prospects</li>
                        <li><b>Diversification:</b> Helps build balanced portfolio</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Company:** {stock_info.get('longName', 'N/A')}")
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                    st.write(f"**Country:** {stock_info.get('country', 'N/A')}")
                    st.write(f"**Employees:** {predictor.safe_format_number(stock_info.get('fullTimeEmployees'))}")
                
                with info_col2:
                    st.write(f"**Website:** {stock_info.get('website', 'N/A')}")
                    st.write(f"**Phone:** {stock_info.get('phone', 'N/A')}")
                    st.write(f"**City:** {stock_info.get('city', 'N/A')}")
                    st.write(f"**State:** {stock_info.get('state', 'N/A')}")
                    st.write(f"**Exchange:** {stock_info.get('exchange', 'N/A')}")
                
                # Business Summary
                if stock_info.get('longBusinessSummary'):
                    st.markdown("### 📝 Business Summary")
                    st.write(stock_info['longBusinessSummary'])
                
                # Download data option
                st.markdown("## 📥 Download Data")
                
                # Download help
                with st.expander("📥 Download Data Help", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>📥 Data Download Options</h4>
                    
                    <h5>📊 Historical Data (CSV)</h5>
                    <p><b>Contains:</b> Date, Open, High, Low, Close, Volume for selected period</p>
                    <p><b>Use for:</b></p>
                    <ul>
                        <li>Personal analysis in Excel/Google Sheets</li>
                        <li>Creating custom charts</li>
                        <li>Backtesting trading strategies</li>
                        <li>Academic research</li>
                    </ul>
                    
                    <h5>📈 Technical Indicators (CSV)</h5>
                    <p><b>Contains:</b> All calculated technical indicators (RSI, MACD, Moving Averages, etc.)</p>
                    <p><b>Use for:</b></p>
                    <ul>
                        <li>Advanced technical analysis</li>
                        <li>Building custom models</li>
                        <li>Educational purposes</li>
                        <li>Strategy development</li>
                    </ul>
                    
                    <h5>💾 File Format</h5>
                    <p>CSV (Comma Separated Values) - Opens in Excel, Google Sheets, and any data analysis software</p>
                    
                    <h5>⚠️ Data Usage Terms</h5>
                    <ul>
                        <li>For personal and educational use only</li>
                        <li>Data is for informational purposes</li>
                        <li>Not for commercial redistribution</li>
                        <li>Always verify data accuracy before making investment decisions</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    csv_data = data.to_csv()
                    st.download_button(
                        label="📊 Download Historical Data (CSV)",
                        data=csv_data,
                        file_name=f"{symbol}_historical_data.csv",
                        mime="text/csv"
                    )
                
                with download_col2:
                    indicators_csv = df_indicators.to_csv()
                    st.download_button(
                        label="📈 Download Technical Indicators (CSV)",
                        data=indicators_csv,
                        file_name=f"{symbol}_technical_indicators.csv",
                        mime="text/csv"
                    )
                
                # Investment Learning Section
                st.markdown("## 📚 Investment Learning Center")
                
                with st.expander("🎓 Learn Investment Basics", expanded=False):
                    st.markdown("""
                    <div class="help-section">
                    <h4>🎓 Investment Education</h4>
                    
                    <h5>📈 Types of Analysis</h5>
                    <ul>
                        <li><b>Technical Analysis:</b> Studies price patterns and indicators</li>
                        <li><b>Fundamental Analysis:</b> Evaluates company's financial health</li>
                        <li><b>Sentiment Analysis:</b> Considers market psychology</li>
                    </ul>
                    
                    <h5>💡 Investment Strategies</h5>
                    <ul>
                        <li><b>Day Trading:</b> Buy/sell within same day</li>
                        <li><b>Swing Trading:</b> Hold for days to weeks</li>
                        <li><b>Position Trading:</b> Hold for months to years</li>
                        <li><b>Buy & Hold:</b> Long-term investment strategy</li>
                    </ul>
                    
                    <h5>⚠️ Risk Management</h5>
                    <ul>
                        <li><b>Diversification:</b> Don't put all eggs in one basket</li>
                        <li><b>Position Sizing:</b> Never risk more than you can afford to lose</li>
                        <li><b>Stop Losses:</b> Set exit points before entering trades</li>
                        <li><b>Research:</b> Always do your own due diligence</li>
                    </ul>
                    
                    <h5>📊 Key Ratios Summary</h5>
                    <ul>
                        <li><b>P/E:</b> Valuation relative to earnings</li>
                        <li><b>P/B:</b> Price vs book value</li>
                        <li><b>ROE:</b> Profitability efficiency</li>
                        <li><b>Debt/Equity:</b> Financial leverage</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer with about app and developer info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #222; margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;'>
        <h3>🚀 About This App</h3>
        <p>
            <b>Complete Share Market Analysis Dashboard</b> is an all-in-one platform for stock analysis, prediction, and learning.<br>
            Built with <b>Streamlit</b> & <b>PyTorch</b> for investors, traders, and learners.
        </p>
        <hr style="margin: 1rem 0;">
        <h4>👨‍💻 Developer Information</h4>
        <p>
            <b>Name:</b> Anil Kumar Singh<br>
            <b>Email:</b> <a href="mailto:singhanil854@gmail.com">singhanil854@gmail.com</a><br>
            <b>GitHub:</b> <a href="https://github.com/anil002" target="_blank">https://github.com/anil002</a>
        </p>
        <hr style="margin: 1rem 0;">
        <p style="font-size: 0.95em;">
            ⚠️ <b>DISCLAIMER:</b> This app is for educational purposes only. Not financial advice.<br>
            📚 Always consult with financial advisors and do your own research before making investment decisions.<br>
            💡 Click the expandable sections above for detailed help and explanations!
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_comprehensive_dashboard()