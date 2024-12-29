import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Function to preprocess data
def preprocess_data(df):
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.reset_index(inplace=True)

    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['Crossover'] = (df['EMA_5'] > df['EMA_21']) & (df['EMA_5'].shift(1) <= df['EMA_21'].shift(1))

    df['Price_Change'] = df['Close'].diff()
    df['Gain'] = df['Price_Change'].where(df['Price_Change'] > 0, 0)
    df['Loss'] = -df['Price_Change'].where(df['Price_Change'] < 0, 0)

    df['Avg_Gain'] = df['Gain'].rolling(window=14, min_periods=1).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=14, min_periods=1).mean()

    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))

    df['RSI_Above_50'] = df['RSI'] >= 50

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)

    return df

# Function to scale features
def scale_features(df, features):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X = df[features]
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled

# Streamlit UI
st.title("Stock Price Prediction App")

# User inputs
ticker_symbol = st.text_input("Enter the Stock Ticker Symbol (e.g., AAPL):", "AAPL")
interval = st.selectbox("Select Interval:", ["5 Minutes", "1 Hour", "1 Week", "15 Days"])

# Prediction button
if st.button("Load Data and Predict"):
    try:
        # Fetch data based on interval
        stock_data = yf.Ticker(ticker_symbol)
        if interval == "5 Minutes":
            df = stock_data.history(period="5d", interval="5m")
            model_file = 'model_5min.pkl'
        elif interval == "1 Hour":
            df = stock_data.history(period="5d", interval="1h")
            model_file = 'model_1hour.pkl'
        elif interval == "1 Week":
            df = stock_data.history(period="1y", interval="1d")
            model_file = 'model_1week.pkl'
        elif interval == "15 Days":
            df = stock_data.history(period="1y", interval="1d")
            model_file = 'model_15days.pkl'

        # Preprocess data
        df = preprocess_data(df)
        features = ['EMA_5', 'EMA_21', 'Crossover', 'RSI', 'RSI_Above_50', 'Price_Change', 'Gain', 'Loss', 'Avg_Gain', 'Avg_Loss', 'RS']
        X_scaled = scale_features(df, features)

        # Load model and predict
        model = joblib.load(model_file)
        prediction = model.predict(X_scaled[-1].reshape(1, -1))

        # Display prediction
        st.subheader(f"Prediction for the next {interval}: {'Price Up' if prediction[0] == 1 else 'Price Down'}")

        # Visualizations
        st.subheader("Stock Data Overview")
        st.line_chart(df[['Close', 'EMA_5', 'EMA_21']])

        st.subheader("RSI Indicator")
        st.line_chart(df['RSI'])

        st.subheader("Price Change Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Price_Change'], kde=True, bins=30)
        st.pyplot(plt)

        # Show latest data points
        st.subheader("Latest Data Points")
        st.dataframe(df.tail(10))

    except Exception as e:
        st.error(f"Error: {e}")
