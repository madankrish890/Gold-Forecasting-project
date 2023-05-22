import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = pickle.load(open('Sarima_model.pkl' , 'rb'))

data=pd.read_csv('Gold_data.csv')

data['date']=pd.to_datetime((data['date']))


def main():
    st.title('Gold Forecasting Deployment')
    st.write('Enter the required parameters and click "Predict" to see the forecast.')

    # Display input fields for the user
    start_date = st.date_input('Start Date', value=data['date'].min())
    end_date = st.date_input('End Date', value=data['date'].max())
    forecast_length = st.number_input('Forecast Length', value=30, min_value=1, max_value=365)

    if st.button('Predict'):
        # Filter data based on selected dates
        filtered_data = data[(data['date'] >= str(start_date)) & (data['date'] <= str(end_date))]

        # Prepare the data for forecasting
        # Assuming your data has 'date' and 'price' columns
        filtered_data = filtered_data[['date', 'price']]
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        filtered_data.set_index('date', inplace=True)

        # Perform forecasting
        forecast = model.get_forecast(steps=forecast_length)

        # Get the predicted values
        predicted_values = forecast.predicted_mean

        # Concatenate the original data and predicted values
        result = pd.concat([filtered_data, predicted_values], axis=1)

        # Plot the results
        st.line_chart(result)

if __name__ == '__main__':
    main()

