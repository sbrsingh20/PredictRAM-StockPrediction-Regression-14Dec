import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import streamlit as st

# Inflation data
data = {
    'Date': ['Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23',
             'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24',
             'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24'],
    'Inflation': [6.155, 6.16, 5.794, 5.09, 4.419, 5.573, 7.544, 6.912, 5.02, 4.87, 5.55, 5.69, 
                  5.1, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49]
}
inflation_df = pd.DataFrame(data)
inflation_df['Date'] = pd.to_datetime(inflation_df['Date'], format='%b-%y')

# Fetch stock data
def fetch_stock_data(ticker):
    stock = yf.download(ticker, start='2023-01-01', end='2024-09-30', interval='1mo')
    stock.reset_index(inplace=True)
    stock['Date'] = stock['Date'].dt.to_period('M').dt.to_timestamp()
    return stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Merge data and train models
def train_model(ticker):
    stock_df = fetch_stock_data(ticker)
    merged_df = pd.merge(stock_df, inflation_df, on='Date')
    
    X = merged_df[['Open', 'High', 'Low', 'Volume', 'Inflation']]
    y = merged_df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = {
        'RandomForest': RandomForestRegressor(),
        'LinearRegression': LinearRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'{ticker} {name}: MSE={mean_squared_error(y_test, y_pred)}, R2={r2_score(y_test, y_pred)}')
        with open(f'{ticker}_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

# Train models for stocks
for stock in ['ITC.NS', 'TCS.NS', 'WIPRO.NS']:
    train_model(stock)

# Streamlit app for prediction
st.title("Stock Price Prediction")
uploaded_file = st.file_uploader("Upload PKL file", type=['pkl'])
if uploaded_file is not None:
    model = pickle.load(uploaded_file)
    inflation_input = st.number_input("Enter Expected Inflation Rate")
    avg_open = st.number_input("Enter Average Open Price")
    avg_high = st.number_input("Enter Average High Price")
    avg_low = st.number_input("Enter Average Low Price")
    avg_volume = st.number_input("Enter Average Volume")

    if st.button("Predict Closing Price"):
        features = np.array([[avg_open, avg_high, avg_low, avg_volume, inflation_input]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Closing Price: {prediction:.2f}")
