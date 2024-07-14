import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Function to load sales data
def load_data(file, date_column, start_date, end_date, date_format=None):
    data = pd.read_csv(file)
    data[date_column] = pd.to_datetime(data[date_column], format=date_format, errors='coerce')
    data = data[(data[date_column] >= start_date) & (data[date_column] <= end_date)]
    data.reset_index(drop=True, inplace=True)
    return data

# Function to create forecast for selected data
def create_forecast(data, date_column, value_column, period, title):
    st.subheader(f'Time Series data for {title}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_column], y=data[value_column], name=title))
    fig.layout.update(title_text=f'Raw Data - {title}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    df_train = data[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})
    if df_train.dropna().shape[0] < 2:
        st.error(f"Not enough data for {title} to create a forecast.")
        return

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader(f'Forecast data for {title}')
    st.write(forecast.tail())

    st.subheader(f'Forecast plot for {title} for {period // 365} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.subheader(f'Forecast components for {title}')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

# Constants
START_DATE = "2017-09-01"
END_DATE = "2019-02-13"

# Paths to the datasets
DATA_PATH1 = "shoe_sales1.csv"
DATA_PATH2 = "nike_shoe_sales.csv"

# Column names
DATE_COLUMN1 = "Order Date"
SNEAKER_NAME_COLUMN1 = "Sneaker Name"
VALUE_COLUMN1 = "Sale Price"
DATE_COLUMN2 = "Order Date"
SNEAKER_NAME_COLUMN2 = "Sneaker Name"
VALUE_COLUMN2 = "Sale Price"

st.title('Sales Forecast Comparison App')

# Load the datasets
data1 = load_data(DATA_PATH1, DATE_COLUMN1, START_DATE, END_DATE)
data2 = load_data(DATA_PATH2, DATE_COLUMN2, START_DATE, END_DATE, date_format="%d-%m-%Y")

# Sidebar for user input
st.sidebar.title('Input Parameters')
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

shoe_types1 = data1[SNEAKER_NAME_COLUMN1].unique()
selected_shoe_type1 = st.sidebar.selectbox('Select Shoe Type from first dataset (Adidas)', shoe_types1)

shoe_types2 = data2[SNEAKER_NAME_COLUMN2].unique()
selected_shoe_type2 = st.sidebar.selectbox('Select Shoe Type from second dataset (Nike)', shoe_types2)

# Filter data based on selected shoe type
filtered_data1 = data1[data1[SNEAKER_NAME_COLUMN1] == selected_shoe_type1]
filtered_data2 = data2[data2[SNEAKER_NAME_COLUMN2] == selected_shoe_type2]

# Display the first few rows of the filtered datasets and their lengths for debugging
st.write(f"Filtered data for {selected_shoe_type1} (Adidas):")
st.write(filtered_data1.head())
st.write(f"Number of rows: {filtered_data1.shape[0]}")

st.write(f"Filtered data for {selected_shoe_type2} (Nike):")
st.write(filtered_data2.head())
st.write(f"Number of rows: {filtered_data2.shape[0]}")

# Columns for side-by-side comparison
col1, col2 = st.columns(2)

# Forecast for the first dataset
with col1:
    st.header(f'Analysis for {selected_shoe_type1} from first dataset (Adidas)')
    create_forecast(filtered_data1, DATE_COLUMN1, VALUE_COLUMN1, period, selected_shoe_type1)

# Forecast for the second dataset
with col2:
    st.header(f'Analysis for {selected_shoe_type2} from second dataset (Nike)')
    create_forecast(filtered_data2, DATE_COLUMN2, VALUE_COLUMN2, period, selected_shoe_type2)