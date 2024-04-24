#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('CleanedDataNike.csv', low_memory=False)


# In[3]:


print(df.info())


# In[4]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[5]:


# Change the data type of the 'SIZE_VALUE' column to categorical
df['SIZE_VALUE'] = df['SIZE_VALUE'].astype('object')

print(df.dtypes)


# In[6]:


#df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'], format='ISO8601', utc=True)
#df['RELEASEDATE'] = pd.to_datetime(df['RELEASEDATE'], format='ISO8601', utc=True)

df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'], utc=True)
df['RELEASEDATE'] = pd.to_datetime(df['RELEASEDATE'], utc=True)


# Verify the changes
print(df.dtypes)


# In[7]:


# Filter the dataset to include data from January 2021 to December 2022
df_filtered = df[(df['SOLD_AT'] >= '2021-01-01') & (df['SOLD_AT'] <= '2022-12-31')]

# Resampling to monthly frequency and averaging 'SOLD_PRICE'
df_resampled = df_filtered.set_index('SOLD_AT').resample('M')['SOLD_PRICE'].mean().dropna()


# In[8]:


# Fit the ARIMA Model
model = ARIMA(df_resampled, order=(1, 1, 1))
model_fit = model.fit()


# In[9]:


# Forecast the next 12 months to cover the year 2023
forecast = model_fit.get_forecast(steps=12)
forecast_index = pd.date_range(start='2023-01-01', periods=12, freq='M')

# Visualization
plt.figure(figsize=(12, 7))
plt.plot(df_resampled.index, df_resampled, label='Historical Monthly Average Sold Price')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecasted Sold Price for 2023', color='red')
plt.fill_between(forecast_index,
                 forecast.conf_int().iloc[:, 0],
                 forecast.conf_int().iloc[:, 1], color='pink', alpha=0.3)
plt.title('Nike Sneaker Resale Price Forecast with ARIMA for 2023')
plt.xlabel('Year-Month')
plt.ylabel('Avg Sold Price $')
plt.legend()
plt.show()


# In[10]:


# Resample to a regular time interval if necessary, e.g., monthly average sold price
df_resampled1 = df.resample('M', on='SOLD_AT')['SOLD_PRICE'].mean().dropna()


# In[11]:


# Fit the ARIMA model
model1 = ARIMA(df_resampled1, order=(1, 1, 1))
model_fit1 = model1.fit()


# In[12]:


# Forecast the next 12 months
forecast1 = model_fit.get_forecast(steps=12)
forecast_index1 = pd.date_range(df_resampled1.index[-1], periods=12, freq='M')

# Adjust the code to add one month to the start date to ensure proper alignment
forecast_index1 = forecast_index1 + pd.offsets.MonthBegin(1)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(df_resampled1.index, df_resampled1, label='Historical Monthly Average Sold Price')
plt.plot(forecast_index1, forecast1.predicted_mean, label='Forecasted Sold Price', color='red')
plt.fill_between(forecast_index1,
                 forecast1.conf_int().iloc[:, 0],
                 forecast1.conf_int().iloc[:, 1], color='pink', alpha=0.3)
plt.title('Nike Sneaker Resale Price Forecast with ARIMA for 2024')
plt.xlabel('Year-Month')
plt.ylabel('Avg Sold Price $')
plt.legend()
plt.show()


# In[ ]:





# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Simulate time series data
np.random.seed(42)
dates_simulated = pd.date_range(start='2021-01-01', end='2022-12-31', freq='M')
data_simulated = np.random.randn(len(dates_simulated)).cumsum() + 200
df_resampled_simulated = pd.Series(data_simulated, index=dates_simulated)

# Plotting code (for reference)
plt.figure(figsize=(14, 6))
plt.subplot(121)
plot_acf(df_resampled_simulated, ax=plt.gca(), title='ACF for Simulated df_resampled', lags=10)
plt.subplot(122)
plot_pacf(df_resampled_simulated, ax=plt.gca(), title='PACF for Simulated df_resampled', lags=10)
plt.tight_layout()
plt.show()


# In[ ]:





# In[14]:


# Resample to monthly frequency, using mean of SOLD_PRICE
df_monthly = df_filtered.set_index('SOLD_AT').resample('M')['SOLD_PRICE'].mean().dropna()

# SARIMA Configuration
p, d, q = 1, 1, 1  # Non-seasonal orders
P, D, Q, S = 1, 1, 1, 12  # Seasonal orders

# Fit the SARIMA Model on the training data
model = SARIMAX(df_monthly, order=(p, d, q), seasonal_order=(P, D, Q, S))
model_fit = model.fit()

# Forecast for 12 months (the entire year of 2023)
forecast = model_fit.get_forecast(steps=12)
forecast_index = pd.date_range(start='2023-01-01', periods=12, freq='M')
forecast_values = forecast.predicted_mean

# Confidence intervals for the forecasts
confidence_intervals = forecast.conf_int()

# Visualization of the Forecast
plt.figure(figsize=(12, 7))
plt.plot(df_monthly.index, df_monthly, label='Historical Monthly Average Sold Price')
plt.plot(forecast_index, forecast_values, label='Forecasted Sold Price with SARIMA for 2023', color='red')
plt.fill_between(forecast_index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.5)
plt.title('Nike Sneaker Resale Price Forecast with SARIMA for 2023')
plt.xlabel('Year-Month')
plt.ylabel('Avg Sold Price $')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


# Resampling to monthly frequency and averaging 'SOLD_PRICE'
df_monthly1 = df.set_index('SOLD_AT').resample('M')['SOLD_PRICE'].mean().dropna()

# Define SARIMA Model Parameters
p, d, q = 1, 1, 1  # Non-seasonal orders
P, D, Q, S = 2, 1, 1, 12  # Seasonal orders (with S=12 for monthly data)

# Adjusting the forecast_index generation without using 'closed'
forecast_index1 = pd.date_range(start=df_monthly1.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='M')

# Fit the SARIMA Model
model1 = SARIMAX(df_monthly1, order=(p, d, q), seasonal_order=(P, D, Q, S))
model_fit1 = model1.fit()

# Forecast the next 12 months
forecast1 = model_fit1.get_forecast(steps=12)

# Corrected forecast_index generation
forecast_index1 = pd.date_range(start=df_monthly1.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='M')

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(df_monthly1.index, df_monthly1, label='Historical Monthly Average Sold Price')
plt.plot(forecast_index1, forecast1.predicted_mean, label='Forecasted Sold Price', color='red')
plt.fill_between(forecast_index1,
                 forecast1.conf_int().iloc[:, 0],
                 forecast1.conf_int().iloc[:, 1], color='pink', alpha=0.3)
plt.title('Nike Sneaker Resale Price Forecast with SARIMA for 2024')
plt.xlabel('Year-Month')
plt.ylabel('Avg Sold Price $')
plt.legend()
plt.show()


# In[ ]:





# In[18]:


# Assuming the length of your dataset is less than or equal to 36
# Adjust the lags to be less than half the size of your dataset
plot_acf(df_monthly, lags=11)  # Adjust lags for ACF as well to maintain consistency
plot_pacf(df_monthly, lags=11)  # Adjust lags for PACF to avoid the error


# Repeat for df_monthly1 if necessary
plot_acf(df_monthly1, lags=17)
plot_pacf(df_monthly1, lags=17)


# In[ ]:





# In[19]:


# Assuming df is your DataFrame with SOLD_AT and SOLD_PRICE
df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'], utc=True)
df.sort_values('SOLD_AT', inplace=True)

# Resample to monthly frequency, using mean of SOLD_PRICE
df_monthly = df.set_index('SOLD_AT').resample('M')['SOLD_PRICE'].mean().dropna()

# Calculate the split index
split_idx = int(len(df_monthly) * 0.8)

# Split the data into training and testing sets
train = df_monthly.iloc[:split_idx]
test = df_monthly.iloc[split_idx:]


# In[20]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def evaluate_model(train, test, order, seasonal_order=None):
    history = train.copy()
    predictions = list()
    
    for t in range(len(test)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # Adjusted to use pd.concat for appending
        history = pd.concat([history, test[t:t+1]])
    
    rmse = sqrt(mean_squared_error(test, predictions))
    
    # Plotting the forecast against the actual outcomes
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, color='gray', label='Actual Value')
    plt.plot(test.index, predictions, color='red', linestyle='--', label='Forecast')
    plt.title('Forecast vs Actuals')
    plt.legend()
    plt.show()
    
    return rmse


# Example ARIMA model evaluation
arima_order = (1, 1, 1)  # adjust based on exploratory analysis
arima_rmse = evaluate_model(train, test, order=arima_order)
print(f"ARIMA RMSE: {arima_rmse}")

# Example SARIMA model evaluation, assuming a placeholder seasonal order
sarima_order = (1, 1, 1, 12)  # Adjust based on dataset
sarima_rmse = evaluate_model(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
print(f"SARIMA RMSE: {sarima_rmse}")


# In[21]:


import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your DataFrame and has been prepared as before
# Splitting and other preparation steps as previously described

def evaluate_model(train, test, order, seasonal_order=None, plot=False):
    history = train.copy()
    predictions = list()
    
    for t in range(len(test)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history = pd.concat([history, test.iloc[[t]]])
    
    rmse = sqrt(mean_squared_error(test, predictions))
    
    if plot:
        # Plotting the forecast against the actual outcomes
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Training Data')
        plt.plot(test.index, test, color='gray', label='Actual Value')
        plt.plot(test.index, predictions, color='red', linestyle='--', label='Forecast')
        plt.title('Forecast vs Actuals')
        plt.legend()
        plt.show()
    
    return rmse

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

def optimize_sarima(train, test, pdq, seasonal_pdq):
    best_score, best_cfg = float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                # Don't plot in the evaluation step
                rmse = evaluate_model(train, test, order=param, seasonal_order=param_seasonal, plot=False)
                if rmse < best_score:
                    best_score, best_cfg = rmse, (param, param_seasonal)
                print('ARIMA{}x{}12 - RMSE:{}'.format(param, param_seasonal, rmse))
            except:
                continue
    # Plot for the best configuration
    evaluate_model(train, test, order=best_cfg[0], seasonal_order=best_cfg[1], plot=True)
    return best_cfg, best_score

# Grid search (this can take a considerable amount of time)
best_cfg, best_score = optimize_sarima(train, test, pdq, seasonal_pdq)
print(f"Best SARIMA{best_cfg} RMSE={best_score}")


# In[ ]:




