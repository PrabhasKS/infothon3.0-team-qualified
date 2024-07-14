import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

DATA_PATH = "mobile1.csv"  # Update this path accordingly
START = "2017-09-01"
TODAY = "2019-02-13"

st.title('Mobile Sales Forecast App')

# Load mobile sales data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns={"State": "state", "Product": "product", "Units Sold": "units_sold"})
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data... done!')

# Sidebar for user input
st.sidebar.title('Input Parameters')
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Search box for mobile products
mobile_products = data['product'].unique()
selected_mobile_product1 = st.sidebar.selectbox('Select First Mobile Product', mobile_products)
selected_mobile_product2 = st.sidebar.selectbox('Select Second Mobile Product', mobile_products)

# Filter data based on selected mobile products
filtered_data1 = data[data['product'] == selected_mobile_product1]
filtered_data2 = data[data['product'] == selected_mobile_product2]

# Prepare time series data
filtered_data1['Date'] = pd.date_range(start=START, periods=len(filtered_data1), freq='D')
df_train1 = filtered_data1[['Date', 'units_sold']].rename(columns={"Date": "ds", "units_sold": "y"})

filtered_data2['Date'] = pd.date_range(start=START, periods=len(filtered_data2), freq='D')
df_train2 = filtered_data2[['Date', 'units_sold']].rename(columns={"Date": "ds", "units_sold": "y"})

# Columns for side-by-side comparison
col1, col2 = st.columns(2)

# Forecast for the first selected mobile product
with col1:
    st.subheader(f'Time Series data for {selected_mobile_product1} with Rangeslider')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train1['ds'], y=df_train1['y'], name=selected_mobile_product1))
    fig.layout.update(title_text=f'Raw Data - {selected_mobile_product1} Sales', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    m1 = Prophet()
    m1.fit(df_train1)
    future1 = m1.make_future_dataframe(periods=period)
    forecast1 = m1.predict(future1)

    st.subheader(f'Forecast data for {selected_mobile_product1}')
    st.write(forecast1.tail())

    st.subheader(f'Forecast plot for {selected_mobile_product1} for {n_years} years')
    fig1 = plot_plotly(m1, forecast1)
    st.plotly_chart(fig1)

    st.subheader(f'Forecast components for {selected_mobile_product1}')
    fig2 = m1.plot_components(forecast1)
    st.write(fig2)

# Forecast for the second selected mobile product
with col2:
    st.subheader(f'Time Series data for {selected_mobile_product2} with Rangeslider')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train2['ds'], y=df_train2['y'], name=selected_mobile_product2))
    fig.layout.update(title_text=f'Raw Data - {selected_mobile_product2} Sales', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    m2 = Prophet()
    m2.fit(df_train2)
    future2 = m2.make_future_dataframe(periods=period)
    forecast2 = m2.predict(future2)

    st.subheader(f'Forecast data for {selected_mobile_product2}')
    st.write(forecast2.tail())

    st.subheader(f'Forecast plot for {selected_mobile_product2} for {n_years} years')
    fig1 = plot_plotly(m2, forecast2)
    st.plotly_chart(fig1)

    st.subheader(f'Forecast components for {selected_mobile_product2}')
    fig2 = m2.plot_components(forecast2)
    st.write(fig2)