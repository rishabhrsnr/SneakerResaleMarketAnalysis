#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[67]:


df = pd.read_csv('cleaned_data.csv')


# In[72]:


# Convert 'SOLD_AT' and 'RELEASEDATE' to datetime format
df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'], format='ISO8601', errors='coerce')

df['RELEASEDATE'] = pd.to_datetime(df['RELEASEDATE'], format='%Y-%m-%d', errors='coerce')


# In[73]:


df.head()


# In[74]:


# Convert 'SIZE_VALUE' to a categorical variable
df['SIZE_VALUE'] = df['SIZE_VALUE'].astype('object')


# In[75]:


# Update 'COLLABORATOR' to 'None' where 'IS_COLLAB' is False
df.loc[df['IS_COLLAB'] == False, 'COLLABORATOR'] = 'None'


# In[76]:


df.info()


# In[77]:


df.isnull().sum()


# In[78]:


# Set 'SOLD_AT' as the index
df.set_index('SOLD_AT', inplace=True)

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

# Drop non-numeric columns from the DataFrame
df_numeric = df.drop(columns=non_numeric_columns)

#extracting target variable
df_target = df['SOLD_PRICE']

# Resample data to monthly frequency and aggregate by mean
df_monthly = df_target.resample('M').mean()


# In[79]:


# Resample the data to get monthly averages
monthly_avg_sold_price = df['SOLD_PRICE'].resample('M').mean()

# Perform seasonal decomposition
decomposition = seasonal_decompose(monthly_avg_sold_price, model='additive', period=12)  # Assuming seasonality of 12 months

# Plot the decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(monthly_avg_sold_price, label='Original', color='blue')
plt.legend(loc='upper left')
plt.title('Monthly Average Sold Price Decomposition')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='green')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='red')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='purple')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[171]:


# Split data into train and test sets (80% train, 20% test)
train_size = int(len(df_monthly) * 0.8)
train, test = df_monthly.iloc[:train_size], df_monthly.iloc[train_size:]


# In[81]:


#ADF Test for stationarity
result = adfuller(train)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
 print('\t%s: %.3f' % (key, value))


# In[112]:


# Perform KPSS test
result = kpss(train)

# Extract and print the test statistic and p-value
test_statistic = result[0]
p_value = result[1]
print(f"Test Statistic: {test_statistic}")
print(f"P-value: {p_value}")


# In[82]:


plot_acf(train)
plot_pacf(train)
plt.show()


# In[115]:


#differencing the data due to kpss
differenced_data = train.diff().dropna()

# Plot ACF and PACF of the differenced series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(differenced_data, ax=ax1)
plot_pacf(differenced_data, ax=ax2)
plt.show()


# In[172]:


# Define function for performance metric
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[173]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import ParameterGrid
import numpy as np


# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 9))
model_fit = model.fit()
            
# Make predictions
train_pred = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
test_pred = model_fit.forecast(steps=len(test))
            
# Calculate performance metrics
rmse_test = calculate_rmse(test.values, test_pred)
print('ARIMA Model RMSE:', rmse_test)
mae_test = calculate_mae(test.values, test_pred)
print('ARIMA Model MAE:', mae_test)
mape_test = calculate_mape(test.values, test_pred)
print('ARIMA Model MAPE:', mape_test)

# Convert index to 'Month_Year' format
train.index = train.index.strftime('%b %Y')
test.index = test.index.strftime('%b %Y')

print(model_fit.summary())   

#Residual plot
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

# Plot monthly forecasts
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Actual (Train)', color='blue')
plt.plot(test.index, test, label='Actual (Test)', color='green')
plt.plot(test.index, test_pred, label='Forecast (Test)', linestyle='--', color='orange')
plt.title('ARIMA Monthly Forecasts')
plt.xlabel('Month_Year')
plt.ylabel('Average Sold Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[174]:


# Split data into train and test sets (80% train, 20% test)
train_size = int(len(df_monthly) * 0.8)
train, test = df_monthly.iloc[:train_size], df_monthly.iloc[train_size:]


# In[175]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt-Winters model
model_hw = ExponentialSmoothing(train, trend='Multiplicative', seasonal='Additive', seasonal_periods=12, initialization_method='estimated')
model_hw_fit = model_hw.fit()

# Make predictions
train_pred_hw = model_hw_fit.fittedvalues
test_pred_hw = model_hw_fit.forecast(steps=len(test))

# Calculate performance metrics
rmse_hw = calculate_rmse(test.values, test_pred_hw)
print('Holtz winter Model RMSE:', rmse_hw)
mae_hw = calculate_mae(test.values, test_pred_hw)
print('Holtz winter Model MAE:', mae_hw)
mape_hw = calculate_mape(test.values, test_pred_hw)
print('Holtz winter Model MAPE:', mape_hw)

# Print summary of the Holt-Winters model
print(model_hw_fit.summary())

# Residual plot
residuals_hw = train - train_pred_hw
plt.figure(figsize=(10, 6))
plt.plot(residuals_hw)
plt.title('Residuals of Holt-Winters Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

# Plot monthly forecasts
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Actual (Train)', color='blue')
plt.plot(test.index, test, label='Actual (Test)', color='green')
plt.plot(test.index, test_pred_hw, label='Forecast (Test)', linestyle='--', color='orange')
plt.title('Holt-Winters Monthly Forecasts')
plt.xlabel('Month_Year')
plt.ylabel('Average Sold Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

