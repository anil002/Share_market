# 🚀 Complete Share Market Analysis Dashboard

A comprehensive AI-powered financial analysis platform that provides professional-grade stock market insights, predictions, and educational content for investors, traders, and learners.

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Architecture](#-technical-architecture)
- [AI Models](#-ai-models)
- [Supported Markets](#-supported-markets)
- [Screenshots](#-screenshots)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)
- [Disclaimer](#-disclaimer)

## ✨ Features

### 🔍 **Smart Stock Search**
- Global company search across multiple exchanges
- Intelligent symbol recognition and matching
- Support for US, Indian, UK, Canadian, and Australian markets
- Auto-detection of exchange suffixes

### 📈 **Advanced Technical Analysis**
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ATR, CCI, MFI
- **Interactive Charts**: Candlestick charts with multiple timeframes
- **Support & Resistance**: Dynamic level calculation and visualization
- **Fibonacci Retracement**: Automatic level plotting
- **Volume Analysis**: Volume-based indicators and patterns

### 🤖 **AI-Powered Predictions**
- **Multiple ML Models**: Random Forest, Gradient Boosting, SVR, LSTM Neural Networks
- **Ensemble Predictions**: Combined model forecasts for higher accuracy
- **Next-Day Price Forecasting**: Real-time prediction with confidence intervals
- **GPU Acceleration**: CUDA support for faster model training

### 🎯 **Smart Trading Signals**
- **Multi-Factor Analysis**: Combines multiple indicators for signal generation
- **Signal Strength**: Strong Buy/Buy/Hold/Sell/Strong Sell recommendations
- **Signal Reasoning**: Detailed explanations for each recommendation
- **Historical Signal Performance**: Track signal accuracy over time

### ⚠️ **Comprehensive Risk Analysis**
- **Volatility Metrics**: Standard deviation, historical volatility
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Maximum Drawdown**: Worst-case scenario analysis
- **Beta Calculation**: Market correlation analysis

### 🏢 **Fundamental Analysis**
- **Valuation Ratios**: P/E, P/B, PEG, Price/Sales, EV/EBITDA
- **Financial Health**: ROE, ROA, Debt/Equity, Current Ratio, Quick Ratio
- **Company Information**: Detailed business profiles and financial data
- **Market Classification**: Market cap categorization and analysis

### 📚 **Educational Content**
- **Built-in Help System**: Comprehensive explanations for all features
- **Investment Learning Center**: Educational content for beginners
- **Interactive Tutorials**: Step-by-step guidance
- **Best Practices**: Risk management and investment strategies

### 📊 **Data Export & Analysis**
- **CSV Downloads**: Historical data and technical indicators
- **Custom Reports**: Formatted analysis reports
- **API Integration**: Easy data access for further analysis

## 🎬 Demo

![Dashboard Overview](https://via.placeholder.com/800x400/667eea/ffffff?text=Dashboard+Overview)

*Live demo available at: [Your Demo URL]*

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for LSTM training)
- GPU (optional, for faster model training)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/anil002/share-market-analysis-dashboard.git
cd share-market-analysis-dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app1.py
```

5. **Open in browser**
```
http://localhost:8501
```

### Docker Installation

```bash
# Build the image
docker build -t stock-dashboard .

# Run the container
docker run -p 8501:8501 stock-dashboard
```

## 📦 Requirements

### Core Dependencies

```txt
streamlit>=1.28.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.3.0
torch>=2.1.0
plotly>=5.15.0
ta>=0.10.0
scipy>=1.10.0
requests>=2.31.0
psutil>=5.9.0
```

### Optional Dependencies

```txt
# For GPU acceleration
torch[cuda]>=2.1.0

# For enhanced visualizations
kaleido>=0.2.1
```

## 🚀 Usage

### Basic Analysis

1. **Search for a stock**
   - Enter company name (e.g., "Apple", "Microsoft")
   - Or use stock symbol (e.g., "AAPL", "MSFT")
   - For Indian stocks: "RELIANCE.NS", "TCS.BO"

2. **Configure analysis**
   - Select time period (1mo to 5y)
   - Choose analysis type (Complete/Technical/Fundamental)
   - Enable advanced features (Fibonacci, S&R levels)

3. **Review results**
   - Key metrics and current price
   - AI predictions with confidence levels
   - Technical analysis charts
   - Trading signals and recommendations

### Advanced Features

#### Custom Model Training
```python
from app1 import ComprehensiveShareMarketPredictor

predictor = ComprehensiveShareMarketPredictor()
data, symbol, error = predictor.fetch_realtime_data("AAPL", "1y")
df_indicators = predictor.train_advanced_models(data)
prediction = predictor.predict_with_ensemble(data)
```

#### Risk Analysis
```python
risk_metrics = predictor.advanced_risk_analysis(data)
print(f"Volatility: {risk_metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
```

#### Trading Signals
```python
signals_df = predictor.calculate_comprehensive_signals(data)
latest_signal = signals_df.iloc[-1]
print(f"Signal: {latest_signal['Signal']}")
```

## 🏗️ Technical Architecture

### System Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Data Pipeline  │    │  ML Engine      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Interactive   │    │ • Yahoo Finance │    │ • Random Forest │
│   Charts        │ ←→ │   API           │ ←→ │ • Gradient Boost│
│ • Real-time     │    │ • Data Cleaning │    │ • LSTM Networks │
│   Updates       │    │ • Feature Eng.  │    │ • Ensemble      │
│ • Export Tools  │    │ • Indicators    │    │ • GPU Support   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Data Ingestion**: Yahoo Finance API → Raw OHLCV data
2. **Feature Engineering**: Technical indicators calculation
3. **Model Training**: Multiple ML models with cross-validation
4. **Prediction**: Ensemble methods for robust forecasting
5. **Visualization**: Interactive Plotly charts and metrics
6. **Export**: CSV downloads and formatted reports

## 🤖 AI Models

### Machine Learning Pipeline

#### 1. **Random Forest Regressor**
- **Purpose**: Robust baseline predictions
- **Features**: 18 technical indicators
- **Strengths**: Handles non-linear relationships, feature importance

#### 2. **Gradient Boosting Regressor**
- **Purpose**: Sequential error correction
- **Features**: Advanced ensemble technique
- **Strengths**: High accuracy, overfitting resistance

#### 3. **LSTM Neural Network**
- **Purpose**: Time series pattern recognition
- **Architecture**: 3-layer LSTM with dropout
- **Strengths**: Long-term dependency modeling

#### 4. **Support Vector Regression**
- **Purpose**: Non-linear pattern detection
- **Kernel**: RBF (Radial Basis Function)
- **Strengths**: High-dimensional data handling

### Model Performance Metrics

- **MSE (Mean Squared Error)**: Prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Variance explanation percentage
- **RMSE**: Root mean squared error in price units

## 🌍 Supported Markets

| Market | Exchange | Suffix | Examples |
|--------|----------|--------|----------|
| 🇺🇸 United States | NASDAQ/NYSE | None | AAPL, MSFT, GOOGL |
| 🇮🇳 India | NSE | .NS | RELIANCE.NS, TCS.NS |
| 🇮🇳 India | BSE | .BO | RELIANCE.BO, TCS.BO |
| 🇬🇧 United Kingdom | LSE | .L | BP.L, VOD.L |
| 🇨🇦 Canada | TSX | .TO | SHOP.TO, RY.TO |
| 🇦🇺 Australia | ASX | .AX | CBA.AX, BHP.AX |

## 📸 Screenshots

### Main Dashboard
![Main Dashboard](https://via.placeholder.com/800x600/667eea/ffffff?text=Main+Dashboard)

### Technical Analysis
![Technical Analysis](https://via.placeholder.com/800x600/f093fb/ffffff?text=Technical+Analysis)

### AI Predictions
![AI Predictions](https://via.placeholder.com/800x600/4facfe/ffffff?text=AI+Predictions)

### Risk Analysis
![Risk Analysis](https://via.placeholder.com/800x600/ff6b6b/ffffff?text=Risk+Analysis)

## 📚 API Documentation

### Core Classes

#### `ComprehensiveShareMarketPredictor`

Main class for stock analysis and prediction.

```python
class ComprehensiveShareMarketPredictor:
    def __init__(self):
        """Initialize the predictor with default settings."""
        
    def fetch_realtime_data(self, symbol: str, period: str) -> Tuple[pd.DataFrame, str, str]:
        """Fetch real-time stock data."""
        
    def train_advanced_models(self, data: pd.DataFrame) -> pd.DataFrame:
        """Train multiple ML models on stock data."""
        
    def predict_with_ensemble(self, data: pd.DataFrame, models: List[str] = None) -> float:
        """Generate ensemble predictions."""
        
    def advanced_risk_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Perform comprehensive risk analysis."""
```

### Key Methods

#### Data Fetching
```python
data, symbol, error = predictor.fetch_realtime_data("AAPL", "1y")
```

#### Technical Analysis
```python
df_indicators = predictor.calculate_advanced_technical_indicators(data)
```

#### Predictions
```python
prediction = predictor.predict_with_ensemble(data, ['random_forest', 'gradient_boosting'])
```

#### Trading Signals
```python
signals = predictor.calculate_comprehensive_signals(data)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all functions
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Support

### Get Help

- 📧 **Email**: [singhanil854@gmail.com](mailto:singhanil854@gmail.com)
- 💻 **GitHub Issues**: [Create an Issue](https://github.com/anil002/share-market-analysis-dashboard/issues)
- 📱 **Discussions**: [GitHub Discussions](https://github.com/anil002/share-market-analysis-dashboard/discussions)

### FAQ

**Q: Is this suitable for beginners?**
A: Yes! The dashboard includes comprehensive educational content and help sections for learning investment concepts.

**Q: Can I use this for live trading?**
A: This tool is for analysis and education only. Always consult financial advisors for investment decisions.

**Q: Does it work with cryptocurrency?**
A: Currently focused on traditional stocks. Crypto support may be added in future versions.

**Q: How accurate are the predictions?**
A: Model accuracy varies by market conditions. Always combine predictions with fundamental analysis.

## 👨‍💻 Developer

**Anil Kumar Singh**
- 📧 Email: [singhanil854@gmail.com](mailto:singhanil854@gmail.com)
- 💻 GitHub: [@anil002](https://github.com/anil002)
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/anil-kumar-singh)

## ⚠️ Disclaimer

**IMPORTANT LEGAL NOTICE**

This application is for educational and informational purposes only. It does not constitute financial advice, investment recommendations, or trading signals. 

### Key Points:

- **Not Financial Advice**: All analysis and predictions are for educational purposes
- **Market Risks**: Stock markets are inherently risky and unpredictable
- **No Guarantees**: Past performance does not guarantee future results
- **Professional Consultation**: Always consult qualified financial advisors
- **Personal Responsibility**: Users are responsible for their own investment decisions
- **Data Accuracy**: While we strive for accuracy, data may contain errors
- **Third-party Data**: We rely on external data sources beyond our control

### Limitation of Liability

The developers and contributors of this project shall not be liable for any financial losses, damages, or consequences arising from the use of this application.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=anil002/share-market-analysis-dashboard&type=Date)](https://star-history.com/#anil002/share-market-analysis-dashboard&Date)

---

<div align="center">

**Built with ❤️ for the investment community**

[⭐ Star this repo](https://github.com/anil002/share-market-analysis-dashboard) • [🐛 Report Bug](https://github.com/anil002/share-market-analysis-dashboard/issues) • [💡 Request Feature](https://github.com/anil002/share-market-analysis-dashboard/issues)

</div>
