import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App")

stocks = ("AAPL", "GOOG", "MSFT", "SBKFF", "^BSESN", "HDB")

selected_stock = st.selectbox("Select the dataset for prediction", stocks)

n_year = st.slider("Years of prediction:", 1, 5)
period = n_year * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
data_load_state = st.text("Loading Data.....")
data = load_data(selected_stock)
data_load_state.text("Loading Data....Done")


num_data_points = st.slider("Number of Data Points to Display", min_value=10, max_value=len(data), value=len(data), step=10)

st.subheader("Raw Data")
st.write(data.tail(num_data_points))

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'][:num_data_points], y=data['Open'][:num_data_points], mode='lines', name='Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'][:num_data_points], y=data['Close'][:num_data_points], mode='lines', name='Close', line=dict(color='red'), showlegend=True))
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price")
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
st.write(forecast_table)

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)


fig1.update_traces(line=dict(color='green'), selector=dict(name='yhat'))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=data['Close'], mode='lines', name='Actual Close', line=dict(color='green'), showlegend=True))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Close', line=dict(color='red'), showlegend=True))

st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)


st.subheader("Predicted Values (From Today)")
today = date.today()
future_df = pd.DataFrame({'ds': pd.date_range(today, periods=365 * n_year)})
forecast_today = m.predict(future_df)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=forecast_today['ds'], y=forecast_today['yhat'], mode='lines', name='Predicted Close', line=dict(color='red'), showlegend=True))
fig3.update_layout(title_text=f"Predicted Close Prices from {today} to {today + pd.DateOffset(days=365 * n_year)}", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig3)
