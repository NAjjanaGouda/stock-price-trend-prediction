import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# App title
st.title("📈 Stock Price Trend Prediction")

# User input
ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Load data
stock = yf.download(
    ticker,
    start="2018-01-01",
    end="2024-01-01",
    progress=False,
    threads=False
)

if stock is None or len(stock) == 0:
    st.warning("⚠️ Data not available right now. Please try again.")
    st.stop()


st.subheader("Stock Data Preview")
st.write(stock.head())

# Plot closing price
st.subheader("Closing Price Trend")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(stock['Close'])
ax.set_title(f"{ticker} Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# Feature engineering
stock['Return'] = stock['Close'].pct_change()
stock['Target'] = np.where(stock['Return'] > 0, 1, 0)
stock.dropna(inplace=True)

X = stock[['Open', 'High', 'Low', 'Close', 'Volume']]
y = stock['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Accuracy
st.subheader("Model Accuracy")
st.write("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
st.write("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Confusion Matrix
st.subheader("Random Forest Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_rf)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# Latest prediction
latest_data = X.iloc[-1].values.reshape(1, -1)
latest_data = scaler.transform(latest_data)
prediction = rf.predict(latest_data)

st.subheader("Latest Prediction")
if prediction[0] == 1:
    st.success("📈 Stock Price Likely to GO UP")
else:
    st.error("📉 Stock Price Likely to GO DOWN")


