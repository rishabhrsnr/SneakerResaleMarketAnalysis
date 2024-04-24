#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pmdarima')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima.utils import ndiffs
from pmdarima import auto_arima
import matplotlib.dates as mdates
import statsmodels.api as sm
import pmdarima as pm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[3]:


df = pd.read_csv('~/Downloads/Newbalance+converse.csv')


# In[4]:


df.head()


# In[5]:


# Preprocess the data
df1 = df.copy(deep = True)
df1['SOLD_AT'] = pd.to_datetime(df1['SOLD_AT'])


# In[6]:


nb_df = df1[df1['BRAND'] == "New Balance"]
converse_df = df1[df1['BRAND'] == 'Converse']


# In[7]:


nb_df_daily = nb_df.resample('D', on='SOLD_AT')['SOLD_PRICE'].mean().reset_index()


# In[8]:


nb_df_daily.head()


# In[9]:


# Trend and Seasonality Analysis
plt.figure(figsize=(10, 6))
plt.plot(nb_df_daily['SOLD_AT'], nb_df_daily['SOLD_PRICE'], marker='o', linestyle='-')
plt.title('Daily Sold Price Trend')
plt.xlabel('Day')
plt.ylabel('Average Sold Price')
plt.grid(True)
plt.show()

# Basic statistics to identify outliers
print(nb_df_daily['SOLD_PRICE'].describe())


# In[10]:


nb_res = seasonal_decompose(nb_df_daily.set_index('SOLD_AT')['SOLD_PRICE'], model='additive', period=7)

# Plotting the decomposed time series
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
nb_res.trend.plot(ax=ax1, title='Trend')
nb_res.seasonal.plot(ax=ax2, title='Seasonality')
nb_res.resid.plot(ax=ax3, title='Residuals')
plt.tight_layout()
plt.show()


# In[11]:


# Splitting the data into train and test sets
train_size = int(len(nb_df_daily) * 0.8)
train, test = nb_df_daily['SOLD_PRICE'][0:train_size], nb_df_daily['SOLD_PRICE'][train_size:len(nb_df_daily)]


# In[12]:


# Choose the number of lags based on the length of the time series, e.g., 20% of the series length or some domain-specific knowledge
number_of_lags = int(len(train) * 0.2)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Adjust the ACF plot
plot_acf(train, ax=axes[0], lags=number_of_lags, alpha=0.05)
axes[0].set_title('Autocorrelation Function')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')

# Adjust the PACF plot
plot_pacf(train, ax=axes[1], lags=number_of_lags, alpha=0.05)
axes[1].set_title('Partial Autocorrelation Function')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()


# In[13]:


sarima_model = auto_arima(train, 
                          start_p=1, start_q=1, 
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=False)


# In[14]:


# Fitting the model
model_fit = sarima_model.fit(train)


# In[15]:


# Forecasting
sarima_forecast = model_fit.predict(n_periods=len(test))


# In[16]:


# Fit Exponential Smoothing with seasonality considered
es_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
es_forecast = es_model.forecast(steps=len(test))


# In[17]:


# Extract the dates for the test set
test_dates = nb_df_daily['SOLD_AT'][train_size:len(nb_df_daily)]


# In[18]:


# Plot the results
plt.figure(figsize=(14,7))
plt.plot(test_dates, test, label='Test')
plt.plot(test_dates, sarima_forecast, label='SARIMA Forecast')
plt.plot(test_dates, es_forecast, label='Exponential Smoothing Forecast')
plt.title('Forecast Comparison')
plt.legend()
plt.show()


# In[19]:


# Calculate errors for SARIMA
mae_sarima = mean_absolute_error(test, sarima_forecast)
rmse_sarima = np.sqrt(mean_squared_error(test, sarima_forecast))
mape_sarima = np.mean(np.abs((test - sarima_forecast) / test)) * 100


# In[20]:


# Calculate errors for Exponential Smoothing
mae_es = mean_absolute_error(test, es_forecast)
rmse_es = np.sqrt(mean_squared_error(test, es_forecast))
mape_es = np.mean(np.abs((test - es_forecast) / test)) * 100


# In[21]:


# Create a DataFrame with the error metrics
error_metrics = {
    'MAE': {'SARIMA': mae_sarima, 'Exponential Smoothing': mae_es},
    'RMSE': {'SARIMA': rmse_sarima, 'Exponential Smoothing': rmse_es},
    'MAPE': {'SARIMA': mape_sarima, 'Exponential Smoothing': mape_es}
}

# Convert the dictionary into a DataFrame
error_df = pd.DataFrame(error_metrics)

# Transpose the DataFrame to have models as columns and metrics as rows
error_df = error_df.T

# Print the DataFrame
print(error_df)


# In[22]:


converse_df_daily = converse_df.resample('D', on='SOLD_AT')['SOLD_PRICE'].mean().reset_index()


# In[23]:


# Trend and Seasonality Analysis
plt.figure(figsize=(10, 6))
plt.plot(converse_df_daily['SOLD_AT'], converse_df_daily['SOLD_PRICE'], marker='o', linestyle='-')
plt.title('Daily Sold Price Trend')
plt.xlabel('Day')
plt.ylabel('Average Sold Price')
plt.grid(True)
plt.show()

# Basic statistics to identify outliers
print(converse_df_daily['SOLD_PRICE'].describe())


# In[24]:


converse_res = seasonal_decompose(converse_df_daily.set_index('SOLD_AT')['SOLD_PRICE'], model='additive', period=365)

# Plotting the decomposed time series
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
converse_res.trend.plot(ax=ax1, title='Trend')
converse_res.seasonal.plot(ax=ax2, title='Seasonality')
converse_res.resid.plot(ax=ax3, title='Residuals')
plt.tight_layout()
plt.show()


# In[25]:


# Splitting the data into train and test sets
train_size = int(len(converse_df_daily) * 0.8)
train, test = converse_df_daily['SOLD_PRICE'][0:train_size], converse_df_daily['SOLD_PRICE'][train_size:len(converse_df_daily)]


# In[26]:


# Choose the number of lags based on the length of the time series, e.g., 20% of the series length or some domain-specific knowledge
number_of_lags = int(len(train) * 0.2)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Adjust the ACF plot
plot_acf(train, ax=axes[0], lags=number_of_lags, alpha=0.05)
axes[0].set_title('Autocorrelation Function')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')

# Adjust the PACF plot
plot_pacf(train, ax=axes[1], lags=number_of_lags, alpha=0.05)
axes[1].set_title('Partial Autocorrelation Function')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()


# In[27]:


sarima_model = auto_arima(train, 
                          start_p=1, start_q=1, 
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=False)


# In[28]:


# Fit SARIMA model with selected order
model_fit = sarima_model.fit(train)


# In[29]:


# Forecasting
sarima_forecast = model_fit.predict(n_periods=len(test))


# In[30]:


# Fit Exponential Smoothing with seasonality considered
es_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
es_forecast = es_model.forecast(steps=len(test))


# In[31]:


# Extract the dates for the test set
test_dates = converse_df_daily['SOLD_AT'][train_size:len(converse_df_daily)]


# In[32]:


# Plot the results
plt.figure(figsize=(14,7))
plt.plot(test_dates, test, label='Test')
plt.plot(test_dates, sarima_forecast, label='SARIMA Forecast')
plt.plot(test_dates, es_forecast, label='Exponential Smoothing Forecast')
plt.title('Forecast Comparison')
plt.legend()
plt.show()


# In[33]:


# Calculate errors for SARIMA
mae_sarima = mean_absolute_error(test, sarima_forecast)
rmse_sarima = np.sqrt(mean_squared_error(test, sarima_forecast))
mape_sarima = np.mean(np.abs((test - sarima_forecast) / test)) * 100


# In[34]:


# Calculate errors for Exponential Smoothing
mae_es = mean_absolute_error(test, es_forecast)
rmse_es = np.sqrt(mean_squared_error(test, es_forecast))
mape_es = np.mean(np.abs((test - es_forecast) / test)) * 100


# In[35]:


# Create a DataFrame with the error metrics
error_metrics = {
    'MAE': {'SARIMA': mae_sarima, 'Exponential Smoothing': mae_es},
    'RMSE': {'SARIMA': rmse_sarima, 'Exponential Smoothing': rmse_es},
    'MAPE': {'SARIMA': mape_sarima, 'Exponential Smoothing': mape_es}
}

# Convert the dictionary into a DataFrame
error_df = pd.DataFrame(error_metrics)

# Transpose the DataFrame to have models as columns and metrics as rows
error_df = error_df.T

# Print the DataFrame
print(error_df)


# In[ ]:




