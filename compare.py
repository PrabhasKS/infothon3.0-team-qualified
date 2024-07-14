import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

DATA_PATH = "shoe_sales1.csv"
START = "2017-09-01"
TODAY = "2019-02-13"

st.title('Shoe Sales Forecast Comparison App')

# Load shoe sales data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data = data[(data['Order Date'] >= START) & (data['Order Date'] <= TODAY)]
    data.reset_index(drop=True, inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data... done!')

# Sidebar for user input
st.sidebar.title('Input Parameters')
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Search boxes for shoe types
st.sidebar.header('Shoe Type Selection')
shoe_types = data['Sneaker Name'].unique()
selected_shoe_type_1 = st.sidebar.selectbox('Select First Shoe Type', shoe_types, key='shoe1')
selected_shoe_type_2 = st.sidebar.selectbox('Select Second Shoe Type', shoe_types, key='shoe2')

# Function to create forecast for selected shoe type
def create_forecast(shoe_type, column):
    filtered_data = data[data['Sneaker Name'] == shoe_type]

    # Plot raw data
    column.subheader(f'Time Series data for {shoe_type} with Rangeslider')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['Order Date'], y=filtered_data['Sale Price'], name=shoe_type))
    fig.layout.update(title_text=f'Raw Data - {shoe_type} Sales', xaxis_rangeslider_visible=True)
    column.plotly_chart(fig)

    # Predict forecast with Prophet
    df_train = filtered_data[['Order Date', 'Sale Price']].rename(columns={"Order Date": "ds", "Sale Price": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display forecast data
    column.subheader(f'Forecast data for {shoe_type}')
    column.write(forecast.tail())

    # Display forecast plot
    column.subheader(f'Forecast plot for {shoe_type} for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    column.plotly_chart(fig1)

    # Display forecast components
    column.subheader(f'Forecast components for {shoe_type}')
    fig2 = m.plot_components(forecast)
    column.write(fig2)

# Columns for side-by-side comparison
col1, col2 = st.columns(2)

# Forecast for the first selected shoe type
with col1:
    st.header(f'Analysis for {selected_shoe_type_1}')
    create_forecast(selected_shoe_type_1, col1)

# Forecast for the second selected shoe type
with col2:
    st.header(f'Analysis for {selected_shoe_type_2}')
    create_forecast(selected_shoe_type_2, col2)