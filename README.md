# 📈 AI-Driven Financial Market Analytics & Prediction Platform

An advanced, production-level fintech dashboard for real-time stock market intelligence, technical analysis, and predictive modeling. This platform leverages Machine Learning and Time-Series Forecasting to provide actionable insights for investors and analysts.

---

## 🚀 Key Features

### 1. **Live Market Intelligence**
- **Real-time Data:** Fetches live stock data using `yfinance`.
- **Advanced Technical Analysis:** Interactive charts with Bollinger Bands, RSI, MACD, ATR, and SMA (20/50).
- **Core Metrics:** Real-time tracking of Price, Volume, Market Cap, and 52-week ranges.

### 2. **AI & Machine Learning Predictions**
- **Price Prediction:** Uses a Random Forest Regressor trained on 10+ engineered features (Momentum, Volatility, Trends).
- **Evaluation Metrics:** Displays professional-grade metrics including **MAE (Mean Absolute Error)**, **RMSE (Root Mean Squared Error)**, and **R² Score**.
- **Feature Importance:** Visualizes the key technical drivers behind the AI's predictions.

### 3. **Time-Series Forecasting (Prophet)**
- **5-Day Projections:** Integrated **Facebook Prophet** model for robust time-series forecasting.
- **Confidence Intervals:** Interactive visualizations showing projected price trends and uncertainty ranges.

### 4. **Portfolio Optimization**
- **Markowitz Optimization:** Implements Modern Portfolio Theory to calculate the **Efficient Frontier**.
- **Asset Allocation:** Suggests optimal weights for a multi-stock portfolio based on the **Maximum Sharpe Ratio**.
- **Interactive Simulation:** Visualizes 2,000 simulated portfolios for risk-vs-reward analysis.

### 5. **Financial Risk Analytics**
- **Risk Metrics:** Calculates **Sortino Ratio**, **Annualized Return**, and **Volatility**.
- **Value at Risk (VaR):** Displays 95% confidence VaR to estimate potential portfolio losses.

### 6. **Sentiment Analysis**
- **Live News Feed:** Fetches recent headlines for any ticker.
- **NLP Sentiment Engine:** Uses **VADER** and **TextBlob** to categorize news as Positive, Neutral, or Negative with subjectivity scoring.

---

## 🛠️ Technology Stack
- **Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Engine:** [yfinance](https://pypi.org/project/yfinance/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
- **Forecasting:** [Facebook Prophet](https://facebook.github.io/prophet/)
- **Visualization:** [Plotly](https://plotly.com/python/)
- **Portfolio Math:** [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/), [SciPy](https://scipy.org/)
- **NLP:** [VADER Sentiment](https://github.com/cjhutto/vaderSentiment), [TextBlob](https://textblob.readthedocs.io/)

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9 or higher installed.

### 2. Clone the Repository
```bash
git clone https://github.com/ErikThiart/ai-stock-dashboard.git
cd ai-stock-dashboard
```

### 3. Setup Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Database Setup (Short Note)
This project **does not require a traditional SQL/NoSQL database**. It utilizes:
- **Live APIs:** Dynamic fetching via `yfinance`.
- **Local Caching:** Uses `@st.cache_data` for high-performance data persistence during an active session, reducing API call overhead and improving responsiveness.

---

## 🖥️ How to Run Locally

Once dependencies are installed, start the dashboard using the following command:

```bash
streamlit run stock_dashboard.py
```

After running, the dashboard will be available at `http://localhost:8501`.

---
