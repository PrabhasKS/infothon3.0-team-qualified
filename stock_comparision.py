import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast Comparison App')

# Columns for side-by-side layout
col1, col2 = st.columns(2)

# Sidebar for Forecast 1
with st.sidebar:
    st.title('Input Parameters for Forecast 1')
    stocks_1 = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock_1 = st.selectbox('Select dataset for prediction 1', stocks_1, key='1')
    n_years_1 = st.slider('Years of prediction for Forecast 1:', 1, 4, key='n_years_1')
    period_1 = n_years_1 * 365

# Sidebar for Forecast 2
with st.sidebar:
    st.title('Input Parameters for Forecast 2')
    stocks_2 = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock_2 = st.selectbox('Select dataset for prediction 2', stocks_2, key='2')
    n_years_2 = st.slider('Years of prediction for Forecast 2:', 1, 4, key='n_years_2')
    period_2 = n_years_2 * 365

# Load data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to create forecast
def create_forecast(df, period, title):
    df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    fig1 = plot_plotly(m, forecast)
    st.write(f'Forecast plot for {title}')
    st.plotly_chart(fig1)

    st.write(f"Forecast components for {title}")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

# Forecast 1
with col1:
    data_load_state_1 = st.text(f'Loading data for {selected_stock_1}...')
    data_1 = load_data(selected_stock_1)
    data_load_state_1.text(f'Loading data for {selected_stock_1}... done!')

    st.header(f'Forecast 1: {selected_stock_1}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data_1['Date'], y=data_1['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    st.header(f'Forecast 1: {n_years_1} years')
    create_forecast(data_1, period_1, f'{selected_stock_1} for {n_years_1} years')

# Forecast 2
with col2:
    data_load_state_2 = st.text(f'Loading data for {selected_stock_2}...')
    data_2 = load_data(selected_stock_2)
    data_load_state_2.text(f'Loading data for {selected_stock_2}... done!')

    st.header(f'Forecast 2: {selected_stock_2}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data_2['Date'], y=data_2['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    st.header(f'Forecast 2: {n_years_2} years')
    create_forecast(data_2, period_2, f'{selected_stock_2} for {n_years_2} years')