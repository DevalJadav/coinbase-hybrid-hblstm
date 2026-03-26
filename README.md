# 📊 Drift-Aware Crypto Forecasting System

A real-time cryptocurrency forecasting system using Bidirectional LSTM (BiLSTM) with drift-aware evaluation, multi-version experimentation, and multi-timeframe predictions (15m, 30m, 1h).

---

## 🔍 Overview

This project focuses on short-term cryptocurrency price forecasting using OHLCV data. The system is designed to operate in a live environment where data continuously evolves, requiring models to adapt to changing market conditions.

The system supports multiple timeframes (15-minute, 30-minute, and 1-hour candles) and evaluates model performance in a rolling, real-time setup.

---

## ⚙️ Key Features

- Real-time forecasting pipeline for crypto assets  
- Multi-timeframe predictions:
  - 15-minute  
  - 30-minute  
  - 1-hour  
- Bidirectional LSTM (BiLSTM) deep learning model  
- Rolling training and incremental updates  
- Drift-aware evaluation using live error tracking  
- Multi-version model experimentation (V1, V2, V3, V4)  
- Comparison against naive baseline  
- Performance metrics:
  - MAE  
  - RMSE  
  - MAPE  
  - Directional Accuracy  
  - Correlation  

---

## 🧠 Model Architecture

- Input Features:
  - OHLCV (Open, High, Low, Close, Volume)
  - Technical indicators (EMA, MA, RSI, MACD)

- Model:
  - Bidirectional LSTM (BiLSTM)
  - Sequence-based learning for time-series forecasting

- Output:
  - Next time-step price prediction for selected timeframe

---

## 🔄 System Workflow

1. **Data Collection**
   - Fetch OHLCV data using Coinbase API  

2. **Preprocessing**
   - Feature engineering (EMA, MA, RSI, MACD)  
   - Sequence generation  

3. **Model Training**
   - Baseline and improved models (V1 → V4)  
   - Rolling window training  

4. **Live Prediction**
   - Predict next candle (15m / 30m / 1h)  
   - Continuous updates  

5. **Evaluation**
   - Compare predictions with actual values  
   - Track error metrics over time  
   - Benchmark against naive baseline  

---

## 📈 Results

Example performance (BTC-USD, 15-minute timeframe):

- MAE: ~130–180  
- RMSE: ~160–260  
- MAPE: ~0.0019  
- Directional Accuracy: ~0.50–0.58  
- Correlation (r): varies across versions  

> Performance varies across timeframes (15m, 30m, 1h) and model versions.

---

## 📊 Example Output

Predicted vs Actual price comparison:

![Prediction vs Actual](results/plot.png)

---

## 🧪 Model Versions

The system includes multiple model iterations:

- **V1** – Baseline implementation  
- **V2** – Improved scaling and training strategy  
- **V3** – Feature-engineered model (EMA, MA, RSI, MACD)  
- **V4** – Advanced version with architectural and training improvements  

Each version is evaluated in a live-like environment using rolling predictions.

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Pandas, NumPy  
- Matplotlib  
- Streamlit (for live dashboard)  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py
