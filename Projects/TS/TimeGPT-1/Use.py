import pandas as pd
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Get your API Key at dashboard.nixtla.io
# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key='nixak-hKCtqmidEZBFo3VDI6f0vn6VTjn3UdvaYpUY3sHjzqzIT1VQBWs6uQ5IXFCWDrLG8Lb7O0RT6t8hvfLU')

# 2. Read historic electricity demand data
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short.csv')

# Ensure 'ds' column is of datetime type
df['ds'] = pd.to_datetime(df['ds'])

# 3. Forecast the next 24 hours
fcst_df = nixtla_client.forecast(df, h=24, level=[80, 90])

# Ensure 'ds' column in forecast DataFrame is also of datetime type
fcst_df['ds'] = pd.to_datetime(fcst_df['ds'])

# 4. Plot your results (optional)
fig, ax = plt.subplots(figsize=(10, 6))
# Use the correct column name for historical data
ax.plot(df['ds'], df['y'], label='Historical')  # Assuming 'ds' is the date/time column and 'y' is the value column
# Use the correct column names for forecast data
ax.plot(fcst_df['ds'], fcst_df['TimeGPT'], label='Forecast', linestyle='--')
ax.fill_between(fcst_df['ds'], fcst_df['TimeGPT-lo-80'], fcst_df['TimeGPT-hi-80'], color='blue', alpha=0.2, label='80% Confidence Interval')
ax.fill_between(fcst_df['ds'], fcst_df['TimeGPT-lo-90'], fcst_df['TimeGPT-hi-90'], color='red', alpha=0.2, label='90% Confidence Interval')

# Set the x-axis to display dates in the format '2016-10-23 08:00:00'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Adjust the interval as needed

ax.set_xlabel('Date/Time')
ax.set_ylabel('Electricity Demand')
ax.set_title('Electricity Demand Forecast')
ax.legend()

# 5. Save the plot as an image
plt.savefig('/data05/wuxinrui/Projects/TS/TimeGPT-1/forecast_plot.png')

# 6. Save the forecast results to a CSV file
fcst_df.to_csv('/data05/wuxinrui/Projects/TS/TimeGPT-1/forecast_results.csv', index=False)

# Show the plot
plt.show()