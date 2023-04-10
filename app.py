import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import yahoo_fin
import yahoo_fin.stock_info as si
import feedparser
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pytz
import streamlit as st
import plotly.figure_factory as ff
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

st.header("Stock Prediction")
country_dropdowns = {
    'USA': ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH','CRM','VZ','V','WBA','WMT','DIS','DOW'],
    'INDIA': ['HDFCBANK.NS','INFY.NS','ICICIBANK.NS','TCS.NS','KOTAKBANK.NS','ITC.NS','LT.NS','HINDUNILVR.NS','AXISBANK.NS','SBIN.NS','BAJFINANCE.NS','BHARTIARTL.NS','ASIANPAINT.NS','TITAN.NS','HCLTECH.NS','TATASTEEL.NS','MARUTI.NS','SUNPHARMA.NS','BAJAJFINSV.NS','M&M.NS','TECHM.NS','TATAMOTORS.NS','POWERGRID.NS','ULTRACEMCO.NS','WIPRO.NS','NTPC.NS','HINDALCO.NS','JSWSTEEL.NS','NESTLEIND.NS','GRASIM.NS','INDUSINDBK.NS','ADANIPORTS.NS','ONGC.NS','DIVISLAB.NS','HDFCLIFE.NS','CIPLA.NS','DRREDDY.NS','TATACONSUM.NS','SBILIFE.NS','BAJAJ-AUTO.NS','APOLLOHOSP.NS','UPL.NS','BRITANNIA.NS','COALINDIA.NS','EICHERMOT.NS','BPCL.NS','SHREECEM.NS','HEROMOTOCO.NS','HDFCBANK.NS'],
    'LONDON': ['MNG.L','PSON.L','BEZ.L','DCC.L','UTG.L','IHG.L','AUTO.L','BRBY.L','SN.L','RTO.L','IAG.L','OCDO.L','JD.L','ADM.L','EDV.L','SBRY.L','CTEC.L','NG.L','BNZL.L','HSX.L','AHT.L','DGE.L','SPX.L','SGE.L','WTB.L','SGRO.L','FRAS.L','CPG.L','HL.L','BA.L','SMT.L','KGF.L','HLMA.L','RIO.L','BME.L','JMAT.L','CRH.L','BARC.L','GSK.L'],
    'SINGAPORE': ['AWG.SG','575.SG','L5I.SG','A04.SG','AWX.SG','1F3.SG','1A4.SG','1AZ.SG','BTY.SG','BRS.SG','BQF.SG','L02.SG','MT1.SG','AZR.SG','Y45.SG','5GI.SG','578.SG','O08.SG','BTF.SG','BKV.SG','5GF.SG','533.SG','BTX.SG','49B.SG','585.SG','5UF.SG','VVL.SG','CLN.SG','U09.SG','5RE.SG','A55.SG','XVG.SG','BQC.SG','AYV.SG','Y35.SG','43Q.SG','5EF.SG','5AU.SG','5UL.SG','HKB.SG','570.SG','L38.SG','541.SG','B9S.SG','43F.SG','505.SG','BBW.SG','5GJ.SG','40F.SG','A34.SG','A31.SG']
}


def show_country_dropdown(country):
    st.write(f"You selected {country}")
    dropdown_options = country_dropdowns[country]
    select_stock = st.selectbox("Select a city", dropdown_options)
    return select_stock

selected_country = st.sidebar.radio("Select a country", ('USA', 'INDIA', 'LONDON', 'SINGAPORE'))


selected_stock = show_country_dropdown(selected_country)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


stock_symbol = selected_stock
start_date = datetime.now(pytz.utc) - timedelta(days=365*10) # 10 years ago
end_date = datetime.now(pytz.utc)
stock_data = si.get_data(stock_symbol, start_date, end_date)


stock_info = yf.Ticker(stock_symbol).info

stock_name = stock_info['longName']


plt.plot(stock_data.index, stock_data['adjclose'])
plt.title(f'{stock_name} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.show()





periods = [50, 100, 200]
for period in periods:
    stock_data[f'{period}d_EMA'] = stock_data['adjclose'].ewm(span=period, adjust=False).mean()



fig1=plt.figure(figsize=(12, 8))
plt.plot(stock_data.index, stock_data['adjclose'], label='Actual')
for period in periods:
    plt.plot(stock_data.index, stock_data[f'{period}d_EMA'], label=f'{period}d EMA')
plt.title(f'{stock_name} Stock Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()
st.pyplot(fig1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['adjclose'].values.reshape(-1, 1))

prediction_days = 30
x_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1, batch_size=1)


# Retrieve news articles for the same time period
rss_feed = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_symbol}&region=US&lang=en-US'
news_articles = []
feed = feedparser.parse(rss_feed)
for entry in feed.entries:
    if start_date <= datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z') <= end_date:
        news_articles.append(entry.title)

# Perform sentiment analysis using TextBlob
sentiment_scores = []
for article in news_articles:
    blob = TextBlob(article)
    sentiment_scores.append(blob.sentiment.polarity)


# Calculate the sentiment analysis
start_prices = stock_data.groupby(stock_data.index.year)['adjclose'].first()
end_prices = stock_data.groupby(stock_data.index.year)['adjclose'].last()
returns = (start_prices - end_prices) / end_prices


# Retrieve news articles for the same time period
rss_feed = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_symbol}&region=US&lang=en-US'
news_articles = []
feed = feedparser.parse(rss_feed)
for entry in feed.entries:
    if start_date <= datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z') <= end_date:
        news_articles.append(entry.title)

# Perform sentiment analysis using TextBlob
sentiment_scores = []
for article in news_articles:
    blob = TextBlob(article)
    sentiment_scores.append(blob.sentiment.polarity)


# Calculate the sentiment analysis
start_prices = stock_data.groupby(stock_data.index.year)['adjclose'].first()
end_prices = stock_data.groupby(stock_data.index.year)['adjclose'].last()
returns = (start_prices - end_prices) / end_prices



stock_data.reset_index(inplace=True)
stock_data['Date'] = stock_data['index']

# Use Prophet to predict the future prices
prophet_df = stock_data.reset_index()[['Date', 'adjclose']].rename({'Date': 'ds', 'adjclose': 'y'}, axis=1)
m = Prophet(daily_seasonality=True)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot the results using Plotly
fig = plot_plotly(m, forecast)
fig.update_layout(title=f'{stock_name} Stock Price Prediction',
                  xaxis_title='Date',
                  yaxis_title='Adjusted Close Price')
fig.show()
st.pyplot(forecast)



