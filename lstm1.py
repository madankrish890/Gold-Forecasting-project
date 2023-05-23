import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model = load_model('lstm_model1.h5')

from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Define function for inverse scaling
def inverse_scale(scaler, data):
    return scaler.inverse_transform(data)

# Define function to make predictions
def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions

# Define function to create dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Load and preprocess the data you want to predict
df = pd.read_csv('Gold_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

data_scaled = preprocess_data(df)
look_back = 3
X, Y = create_dataset(data_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Create Streamlit app
def main():
    st.title("LSTM Model Deployment")
    st.write('Enter the required parameters and click "Predict" to see the forecast.')

    start_date = st.date_input('Start Date', value=df.index.min())
    end_date = st.date_input('End Date', value=df.index.max())
    forecast_length = st.number_input('Forecast Length (days)', value=30, min_value=1, max_value=365)

    if st.button('Predict'):
        # Filter the data based on selected dates
        filtered_data = df.loc[start_date:end_date]

        # Preprocess the filtered data
        data_scaled = preprocess_data(filtered_data)
        X, Y = create_dataset(data_scaled, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Make predictions
        predictions = make_predictions(model, X)

        # Inverse scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(filtered_data)
        predictions = inverse_scale(scaler, predictions)

        # Get the last 6 months of actual and predicted values
        last_6_months = filtered_data.tail(180)
        predicted_6_months = pd.DataFrame(predictions[-180:], index=last_6_months.index, columns=['Predicted'])

        # Concatenate actual and predicted values
        combined_data = pd.concat([last_6_months, predicted_6_months], axis=1)

        # Display the line chart
        st.line_chart(combined_data)

        # Display the actual and predicted values in a table
        st.subheader("Actual and Predicted Values")
        st.write(combined_data)

        # Make predictions
        predictions = make_predictions(model, X)

        # Inverse scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(filtered_data)
        predictions = inverse_scale(scaler, predictions)

        # Get the last 6 months of actual values
        last_6_months_actual = filtered_data.tail(6 * 30)

        # Get the forecast dates for the next 6 months
        forecast_dates = pd.date_range(start=last_6_months_actual.index[-1] + pd.DateOffset(days=1),
                                       periods=forecast_length)

        # Create a DataFrame for the forecasted values
        forecast_df = pd.DataFrame(index=forecast_dates, columns=['Predicted'])

        # Assign the predicted values to the forecast period
        forecast_df['Predicted'] = predictions[-forecast_length:].flatten()

        # Combine the actual, predicted, and last 6 months data
        combined_data = pd.concat([last_6_months_actual, forecast_df])

        # Display the line chart of actual, predicted, and last 6 months values
        st.line_chart(combined_data)

        # Display the table of actual, predicted, and last 6 months values
        st.subheader("Actual vs Predicted - Last 6 Months + Forecast")

if __name__ == '__main__':
    main()
