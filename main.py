import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

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

st.subheader("Raw Data")
st.write(data.tail())
print("TODAY:", TODAY)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)


plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

