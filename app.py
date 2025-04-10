import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from datetime import timedelta
import os
from io import BytesIO
import time  # For timestamp-based filenames

# Ensure directories exist
test_dir = "test"
forecast_dir = "forecast"
comparison_dir = "comparison"
for directory in [test_dir, forecast_dir, comparison_dir]:
    os.makedirs(directory, exist_ok=True)

# Function to convert wide to long format
def wide_to_long(df, start_col=2, periods_per_day=48):
    try:
        time_series = []
        timestamps = []
        for _, row in df.iterrows():
            date_val = pd.to_datetime(row['Date'])
            for j in range(1, periods_per_day + 1):
                half_hour_offset = pd.Timedelta(minutes=30 * (j - 1))
                ts = date_val + half_hour_offset
                consumption_val = row[f'kwh_{j}']
                timestamps.append(ts)
                time_series.append(consumption_val)
        return pd.DataFrame({'Consumption': time_series}, index=pd.to_datetime(timestamps))
    except Exception as e:
        st.error(f"Error in wide_to_long: {e}")
        return None

# Function to reshape long to wide with total
def reshape_to_wide_with_total(df, periods_per_day=48, consumption_col='Predicted_Consumption'):
    try:
        result = []
        for date, group in df.groupby(df.index.date):
            if len(group) == periods_per_day:
                row = {'Date': pd.Timestamp(date).strftime('%Y-%m-%d %H:%M:%S'), 'Measurement': 'AI'}
                daily_total = 0
                for i, value in enumerate(group[consumption_col], 1):
                    row[f'kwh_{i}'] = value
                    daily_total += value
                row['Total'] = daily_total
                result.append(row)
        return pd.DataFrame(result)
    except Exception as e:
        st.error(f"Error in reshape_to_wide_with_total: {e}")
        return None

# Forecasting function with test file saving
def forecast_energy(file, filename):
    try:
        st.write("Loading and processing file...")
        df = pd.read_excel(file, skiprows=7)
        num_columns = len(df.columns)
        df.columns = ['Date', 'Measurement'] + [f'kwh_{i}' for i in range(1, num_columns - 1)]
        df = df.iloc[:, :50].drop(columns=['kwh_49', 'kwh_50'], errors='ignore')
        df['Date'] = pd.to_datetime(df['Date'])

        ts_df = wide_to_long(df)
        if ts_df is None:
            return None, None, None
        ts_df.sort_index(inplace=True)

        # Feature engineering
        st.write("Performing feature engineering...")
        ts_df['day_of_week'] = ts_df.index.dayofweek
        ts_df['hour'] = ts_df.index.hour
        ts_df['minute'] = ts_df.index.minute
        ts_df['day_of_year'] = ts_df.index.dayofyear
        ts_df['month'] = ts_df.index.month
        ts_df['hour_sin'] = np.sin(2 * np.pi * ts_df['hour'] / 24.0)
        ts_df['hour_cos'] = np.cos(2 * np.pi * ts_df['hour'] / 24.0)
        ts_df['dow_sin'] = np.sin(2 * np.pi * ts_df['day_of_week'] / 7.0)
        ts_df['dow_cos'] = np.cos(2 * np.pi * ts_df['day_of_week'] / 7.0)
        ts_df['lag_1'] = ts_df['Consumption'].shift(1)
        ts_df['lag_2'] = ts_df['Consumption'].shift(2)
        ts_df['lag_48'] = ts_df['Consumption'].shift(48)
        ts_df['lag_96'] = ts_df['Consumption'].shift(96)
        ts_df['rolling_mean_48'] = ts_df['Consumption'].rolling(window=48).mean()
        ts_df['rolling_mean_336'] = ts_df['Consumption'].rolling(window=336).mean()
        ts_df['rolling_std_48'] = ts_df['Consumption'].rolling(window=48).std()

        full_ts_df = ts_df.copy()
        ts_df.dropna(inplace=True)

        feature_cols = [
            'day_of_year', 'month', 'minute', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'lag_1', 'lag_2', 'lag_48', 'lag_96', 'rolling_mean_48', 'rolling_mean_336', 'rolling_std_48'
        ]

        X = ts_df[feature_cols]
        y = ts_df['Consumption']

        total_rows = len(X)
        rows_per_day = 48
        train_size = int((total_rows // rows_per_day) * 0.8 * rows_per_day)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        st.write("Training XGBoost model...")
        tscv = TimeSeriesSplit(n_splits=3)
        param_dist = {
            'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5],
            'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0], 'reg_alpha': [0, 0.5],
            'reg_lambda': [1, 2]
        }
        xgb_model = xgb.XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=tscv, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Test predictions
        st.write("Generating test predictions...")
        test_pred = best_model.predict(X_test)
        test_df = pd.DataFrame({'Predicted_Consumption': test_pred}, index=X_test.index)
        wide_test = reshape_to_wide_with_total(test_df)

        # Extract identifier and date from filename
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split()
        identifier = next((part for part in parts if part.isdigit() and len(part) > 5), 'unknown')
        date_part = next((part for part in parts if '4_9_2025' in part), 'unknown_date').replace('(original)', '')
        test_filename = f"test_predictions_{identifier}_{date_part}.csv"
        test_file_path = os.path.join(test_dir, test_filename)
        wide_test.to_csv(test_file_path, index=False)
        st.write(f"Test predictions saved to {test_file_path}")

        # Forecast 90 days
        st.write("Generating 90-day forecast...")
        last_date = full_ts_df.index.max()
        forecast_start_date = last_date + timedelta(days=1)
        forecast_dates = [forecast_start_date + timedelta(days=i) for i in range(90)]
        forecast_timestamps = [d + timedelta(minutes=30 * i) for d in forecast_dates for i in range(48)]
        forecast_df = pd.DataFrame(index=pd.to_datetime(forecast_timestamps))
        forecast_df['Predicted_Consumption'] = 0.0

        last_row = full_ts_df[feature_cols].iloc[-1].copy()
        last_consumption = full_ts_df['Consumption'].iloc[-96:].values
        forecast_X = []

        for i in range(90 * 48):
            pred = best_model.predict(last_row[feature_cols].values.reshape(1, -1))[0]
            forecast_X.append(pred)
            timestamp = forecast_df.index[i]
            new_row = pd.Series(0.0, index=feature_cols)
            new_row['day_of_week'] = timestamp.dayofweek
            new_row['hour'] = timestamp.hour
            new_row['minute'] = timestamp.minute
            new_row['day_of_year'] = timestamp.dayofyear
            new_row['month'] = timestamp.month
            new_row['hour_sin'] = np.sin(2 * np.pi * new_row['hour'] / 24.0)
            new_row['hour_cos'] = np.cos(2 * np.pi * new_row['hour'] / 24.0)
            new_row['dow_sin'] = np.sin(2 * np.pi * new_row['day_of_week'] / 7.0)
            new_row['dow_cos'] = np.cos(2 * np.pi * new_row['day_of_week'] / 7.0)
            new_row['lag_1'] = pred if i > 0 else last_consumption[-1]
            new_row['lag_2'] = forecast_X[-1] if i > 0 else last_consumption[-2]
            new_row['lag_48'] = forecast_X[-48] if i >= 47 else last_consumption[-48 + i]
            new_row['lag_96'] = forecast_X[-96] if i >= 95 else last_consumption[-96 + i]
            new_row['rolling_mean_48'] = np.mean(forecast_X[-47:i+1]) if i >= 47 else np.mean(np.concatenate([last_consumption[-48+i:], forecast_X[:i+1]]))
            new_row['rolling_mean_336'] = np.mean(forecast_X[-335:i+1]) if i >= 335 else np.mean(np.concatenate([last_consumption[-336+i:], forecast_X[:i+1]]))
            new_row['rolling_std_48'] = np.std(forecast_X[-47:i+1]) if i >= 47 else np.std(np.concatenate([last_consumption[-48+i:], forecast_X[:i+1]]))
            last_row = new_row

        forecast_df['Predicted_Consumption'] = forecast_X
        wide_forecast = reshape_to_wide_with_total(forecast_df)
        forecast_filename = f"forecast_90_days_{identifier}_{date_part}.csv"
        forecast_file_path = os.path.join(forecast_dir, forecast_filename)
        wide_forecast.to_csv(forecast_file_path, index=False)
        st.write(f"Forecast saved to {forecast_file_path}")

        return wide_forecast, forecast_df, wide_test
    except Exception as e:
        st.error(f"Error in forecast_energy: {e}")
        return None, None, None

# Comparison function
def compare_actual_predicted(actual_file, predicted_file):
    try:
        st.write("Loading actual data...")
        actual_df = pd.read_csv(actual_file, skiprows=7)
        num_columns = len(actual_df.columns)
        actual_df.columns = ['Date', 'Measurement'] + [f'kwh_{i}' for i in range(1, num_columns - 1)]
        actual_df = actual_df.iloc[:, :50].drop(columns=['kwh_49', 'kwh_50'], errors='ignore')
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        actual_ts_df = wide_to_long(actual_df)
        if actual_ts_df is None:
            return None, None
        actual_ts_df.columns = ['Actual_Consumption']

        st.write("Loading predicted data...")
        predicted_df = pd.read_csv(predicted_file)
        predicted_ts_df = wide_to_long(predicted_df)
        if predicted_ts_df is None:
            return None, None
        predicted_ts_df.columns = ['Predicted_Consumption']

        st.write("Merging data...")
        comparison_df = actual_ts_df.join(predicted_ts_df, how='inner')
        comparison_df.dropna(inplace=True)

        if len(comparison_df) > 0:
            rmse = np.sqrt(mean_squared_error(comparison_df['Actual_Consumption'], comparison_df['Predicted_Consumption']))
            mae = mean_absolute_error(comparison_df['Actual_Consumption'], comparison_df['Predicted_Consumption'])
            r2 = r2_score(comparison_df['Actual_Consumption'], comparison_df['Predicted_Consumption'])
            mape = np.mean(np.abs((comparison_df['Actual_Consumption'] - comparison_df['Predicted_Consumption']) / comparison_df['Actual_Consumption'])) * 100
            return comparison_df, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        return comparison_df, None
    except Exception as e:
        st.error(f"Error in compare_actual_predicted: {e}")
        return None, None

# Streamlit app
st.title("Energy Consumption Forecasting and Comparison")
st.write("Welcome to the app!")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Forecast Energy Consumption", "Compare Actual vs Predicted"])

if page == "Forecast Energy Consumption":
    st.header("Forecast Energy Consumption")
    uploaded_file = st.file_uploader("Upload historical data (Excel file)", type=["xlsx"])
    
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        if st.button("Generate Forecast"):
            with st.spinner("Training model and generating forecast..."):
                wide_forecast, forecast_df, wide_test = forecast_energy(uploaded_file, uploaded_file.name)
                if wide_forecast is not None and forecast_df is not None and wide_test is not None:
                    st.success("Forecast and test data generated successfully!")
                    
                    st.subheader("90-Day Forecast")
                    st.dataframe(wide_forecast)
                    
                    st.subheader("Test Predictions")
                    st.dataframe(wide_test)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(forecast_df.index, forecast_df['Predicted_Consumption'], label='Forecast', color='blue')
                    ax.set_title("XGBoost Forecast for Next 90 Days")
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("Energy Consumption (kWh)")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Download forecast CSV
                    csv_buffer = BytesIO()
                    wide_forecast.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    forecast_filename = f"forecast_90_days_{os.path.splitext(uploaded_file.name)[0].split()[-2]}_{os.path.splitext(uploaded_file.name)[0].split()[-1].replace('(original)', '')}.csv"
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv_buffer,
                        file_name=forecast_filename,
                        mime="text/csv"
                    )
                    
                    # Download test CSV
                    test_csv_buffer = BytesIO()
                    wide_test.to_csv(test_csv_buffer, index=False)
                    test_csv_buffer.seek(0)
                    test_filename = f"test_predictions_{os.path.splitext(uploaded_file.name)[0].split()[-2]}_{os.path.splitext(uploaded_file.name)[0].split()[-1].replace('(original)', '')}.csv"
                    st.download_button(
                        label="Download Test CSV",
                        data=test_csv_buffer,
                        file_name=test_filename,
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to generate forecast or test data. Check the error messages above.")

elif page == "Compare Actual vs Predicted":
    st.header("Compare Actual vs Predicted Consumption")
    actual_file = st.file_uploader("Upload actual data (CSV file)", type=["csv"], key="actual")
    predicted_file = st.file_uploader("Upload predicted data (CSV file)", type=["csv"], key="predicted")
    
    if actual_file is not None and predicted_file is not None:
        st.write("Both files uploaded successfully!")
        if st.button("Compare Data"):
            with st.spinner("Comparing actual and predicted data..."):
                comparison_df, metrics = compare_actual_predicted(actual_file, predicted_file)
                if comparison_df is not None:
                    st.success("Comparison completed!")
                    
                    st.subheader("Comparison Table")
                    st.dataframe(comparison_df)
                    
                    if metrics:
                        st.subheader("Evaluation Metrics")
                        st.write(f"RMSE: {metrics['RMSE']:.3f} kWh")
                        st.write(f"MAE: {metrics['MAE']:.3f} kWh")
                        st.write(f"R-squared: {metrics['R2']:.3f}")
                        st.write(f"MAPE: {metrics['MAPE']:.2f}%")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(comparison_df.index, comparison_df['Actual_Consumption'], label='Actual', color='blue')
                    ax.plot(comparison_df.index, comparison_df['Predicted_Consumption'], label='Predicted', color='red')
                    ax.set_title("Actual vs Predicted Energy Consumption")
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("Energy Consumption (kWh)")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Save and download comparison CSV
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    comparison_filename = f"actual_vs_predicted_comparison_{timestamp}.csv"
                    comparison_file_path = os.path.join(comparison_dir, comparison_filename)
                    comparison_df.to_csv(comparison_file_path, index=True)
                    st.write(f"Comparison saved to {comparison_file_path}")
                    
                    csv_buffer = BytesIO()
                    comparison_df.to_csv(csv_buffer, index=True)
                    csv_buffer.seek(0)
                    st.download_button(
                        label="Download Comparison CSV",
                        data=csv_buffer,
                        file_name=comparison_filename,
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to compare data. Check the error messages above.")
    else:
        st.write("Please upload both actual and predicted data files.")