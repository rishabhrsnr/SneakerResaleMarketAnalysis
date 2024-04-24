#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Author: Rishabh Bansal
lastRevisionDate: 03-25-2024

Purpose of this Python Script:

1. Data Cleaning:
   - To ensure the integrity and quality of the sneaker resale market dataset by addressing missing values, removing duplicates, and enforcing data type consistency.

2. Exploratory Data Analysis (EDA):
   - To perform a preliminary examination of the dataset to understand the distribution of key variables, identify outliers, and discover patterns that could inform subsequent predictive modeling.

3. Predictive Modeling:
   - To create and refine predictive models that estimate future sneaker prices based on historical data, using time series analysis techniques such as SARIMA.

4. Data Visualization:
   - To translate the analytical results into visual formats, making the insights accessible and understandable. Visualizations include histogram distributions of sneaker prices, boxplots to show price ranges, and time series decomposition plots for trend analysis.

The script is a comprehensive toolkit for processing and analyzing sneaker resale data, aiming to draw actionable insights on pricing trends and the factors influencing them.

Note: This script should be used as part of a larger data analytics process, where the findings here can inform business decisions and strategies in the sneaker resale market.
"""



# In[2]:


import pandas as pd
df0=pd.read_csv('adidas.csv')
# Calculate the number of null values in each column of DataFrame df0
null_counts= df0.isnull().sum()
print
(null_counts)


# In[3]:


# Calculate the percentage of null values for each column
null_percentage=(df0.isnull().sum()/ len(df0))* 100
# Create a DataFrame to display the results
null_percentage_df= pd.DataFrame({
    'Column': null_percentage.index,
 'Null Percentage': null_percentage.values.round(2)
})
print(null_percentage_df)


# In[4]:


df = df0.dropna(subset=['SKU', 'NAME', 'SIZE_VALUE', 'RELEASEDATE', 'COLORWAY', 'SILHOUETTE'], inplace=True)
#Dropping unecessary column
df = df0.drop(columns=['SIZE'])
# Removing duplicates
df = df.drop_duplicates()

# Try parsing with timezone information
df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'], format='ISO8601', errors='coerce')
df['RELEASEDATE'] = pd.to_datetime(df['RELEASEDATE'], format='%Y-%m-%d', errors='coerce')
# Convert the 'IS_COLLAB' column in the DataFrame 'df' to boolean
df['IS_COLLAB'] = df['IS_COLLAB'].astype(bool)

# Update 'COLLABORATOR' to 'None' where 'IS_COLLAB' is False
df.loc[df['IS_COLLAB'] == False, 'COLLABORATOR'] = 'None'
# Count null values in 'COLLABORATOR' where 'IS_COLLAB' is True
null_collaborator_count = df.loc[df['IS_COLLAB'] == 1, 'COLLABORATOR'].isnull().sum()
# Print the result
print(f"Number of null values in 'COLLABORATOR' where 'IS_COLLAB' is True: {null_collaborator_count}")
# Identify rows with null "COLLABORATOR"
null_collaborator_rows = df[df['COLLABORATOR'].isnull()]
# Generate distinct values for "COLLABORATOR" based on "SKU"
distinct_collaborator_values = null_collaborator_rows['SKU'].unique()


# In[5]:


# Replace null "COLLABORATOR" values with distinct "SKU" values
df.loc[df['COLLABORATOR'].isnull(), 'COLLABORATOR'] = df.loc[df['COLLABORATOR'].isnull(), 'SKU'].apply(lambda x:distinct_collaborator_values[0])

# Count null values in 'COLLABORATOR' where 'IS_COLLAB' is True
null_collaborator_count = df.loc[df['IS_COLLAB'] == 1, 'COLLABORATOR'].isnull().sum()
# Print the result
print(f"Number of null values in 'COLLABORATOR' where 'IS_COLLAB' is True: {null_collaborator_count}")
pairs_count = df.groupby(['SIZE_VALUE', 'GENDER']).size().reset_index(name='count')
print(pairs_count)


# In[6]:


#removes rows from the DataFrame where the 'SIZE_VALUE' column contains values greater than 16.0.
df.drop(df[df['SIZE_VALUE'] > 16.0].index, inplace=True)


# In[7]:


# Run the code to change the gender
child_indices = df[(df['SIZE_VALUE'] < 6) & (df['GENDER'] == 'men')].index
df.loc[child_indices, 'GENDER'] = 'child'


# In[8]:


# Run the code to change the gender
child_indices = df[(df['SIZE_VALUE'] < 6) & (df['GENDER'] == 'unisex')].index
df.loc[child_indices, 'GENDER'] = 'child'


# In[9]:


# Run the code to change the gender
child_indices = df[(df['SIZE_VALUE'] < 4) & (df['GENDER'] == 'women')].index
df.loc[child_indices, 'GENDER'] = 'child'


# In[10]:


# Convert 'SIZE_VALUE' to a categorical variable
df['SIZE_VALUE'] = df['SIZE_VALUE'].astype('category')


# In[11]:


df.isnull().sum()


# In[ ]:


df.to_csv('data.csv', index=False)


# In[ ]:


df1=pd.read_csv('data.csv')
df1


# In[16]:


df1.isnull().sum()


# 

# In[26]:


df.info()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# This code assumes the DataFrame 'df' is already loaded in your Jupyter Notebook environment.

# Distribution of sold prices with improved aesthetics
plt.figure(figsize=(10, 6))

sns.histplot(df['SOLD_PRICE'], kde=True, bins=50, color='skyblue').set(xlim=(0, 2500))

plt.title('Adidas Sneaker Sold Prices Distribution', fontsize=16, weight='bold')
plt.xlabel('Sold Price ($)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
# Tight layout to ensure the labels and titles fit into the figure cleanly
plt.tight_layout()
plt.show()


# In[19]:


# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# This code assumes the DataFrame 'df' is already loaded in your Jupyter Notebook environment.

# Boxplot to visualize outliers and distribution characteristics
plt.figure(figsize=(10, 6))
# Adjust the x-axis to focus on the interquartile range of the data
# Set whiskers to 1.5 * IQR
sns.boxplot(x=df['SOLD_PRICE'], showfliers=True, color='lightblue', whiskerprops={'linewidth':2})
plt.title('Boxplot of Adidas Sneaker Sold Prices', fontsize=16, weight='bold')
plt.xlabel('Sold Price ($)', fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.xlim(-100, 2500)  # Adjust this based on your data to exclude extreme outliers for better visualization
# Adding mean price as a distinct marker
mean_price = df['SOLD_PRICE'].mean()
plt.axvline(mean_price, color='r', linestyle='--')
plt.text(mean_price, 0.95, 'Mean', transform=plt.gca().get_xaxis_transform(), ha='center', va='center',
         fontsize=10, color='r', weight='bold')

# Tight layout to ensure the labels and titles fit into the figure cleanly
plt.tight_layout()
plt.show()


# In[20]:


# Time Series Analysis for Adidas brand

# Ensure SOLD_AT is in datetime format
df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'])

# Group by SOLD_AT date and calculate the mean of SOLD_PRICE
df_time_series = df.set_index('SOLD_AT').resample('M')['SOLD_PRICE'].mean()

# Decompose the time series to observe trends and seasonality
decomposition = seasonal_decompose(df_time_series.dropna(), model='additive')

# Plotting the decomposed time series components
decomposition.plot()
plt.show()

# Fit SARIMA model for forecasting
# to determine the best parameters (p, d, q) and seasonal parameters (P, D, Q, s)
model = SARIMAX(df_time_series.dropna(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

# Predict future values
forecast = results.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df_time_series.index, df_time_series, label='Observed')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Adidas Sneaker Sold Prices Forecast')
plt.xlabel('Date')
plt.ylabel('Mean Sold Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Display the summary of the SARIMA model fit
results.summary()


# In[12]:


df['SOLD_AT'] = pd.to_datetime(df['SOLD_AT'], format='date_format_string', errors='coerce')
df.set_index('SOLD_AT', inplace=True)


# In[25]:


print(df.columns)


# In[12]:


import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from datetime import timedelta





# Split the data into a training and a test set
# Let's say you want to use 80% of the data for training and the rest for testing
split_ratio = 0.8
split_index = int(split_ratio * len(df))
train = df.iloc[:split_index]['SOLD_PRICE']
test = df.iloc[split_index:]['SOLD_PRICE']

# Ensure there are no NaN values, which can interfere with ACF and PACF calculations
train = train.dropna()

# Plot ACF and PACF on the training data
fig, axes = plt.subplots(1, 2, figsize=(16, 3))

# Plot the ACF
plot_acf(train, ax=axes[0], lags=40)  # You can adjust the number of lags as needed

# Plot the PACF
plot_pacf(train, ax=axes[1], lags=40)  # You can adjust the number of lags as needed

# Show plots
plt.show()


# In[1]:


jupyter notebook --generate-config


# In[5]:


import sys
sys.setrecursionlimit(10000)
print(sys.getrecursionlimit())


# In[14]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assuming we've already split the data into 'train' and 'test'
# and the 'train' variable contains the training data

# SARIMA Model Order
# p: AR order (based on PACF)
# d: Differencing (assuming 1 to make the series stationary)
# q: MA order (based on ACF)
# P, D, Q, s: Seasonal components (if known, otherwise can be set to 0 or determined via domain knowledge)

# Let's start with a simple non-seasonal SARIMA model, assuming no clear seasonality
p = 1  # AR order
d = 1  # Differencing
q = 0  # MA order

# Fit the SARIMA model
sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)

# Diagnostics plot
sarima_result.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecast
forecast_steps = len(test)  # Number of steps to forecast is the size of the test set
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_df = forecast.conf_int()
forecast_df['forecast'] = forecast.predicted_mean

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.show()

# Calculate the Mean Squared Error on the test set
mse = mean_squared_error(test, forecast_df['forecast'])
print(f"The Mean Squared Error of the forecasts is {mse:.2f}")


# In[ ]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import qqplot

# Use seaborn style for professional quality plots
sns.set_style("whitegrid")


# SARIMA Model (Adjust the order and seasonal_order based on your dataset)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
sarima_result = sarima_model.fit()

# Diagnostic plot for residuals
residuals = sarima_result.resid

# Plot the residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals, color='blue')
plt.title('Residuals from SARIMA model', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Residual', fontsize=14)
plt.show()


# Q-Q plot for residuals
qqplot(residuals, line='s', ax=plt.gca())
plt.title('Q-Q Plot of Residuals', fontsize=16)
plt.show()

# Histogram of residuals
plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Distribution of Residuals', fontsize=16)
plt.xlabel('Residual', fontsize=14)
plt.show()

# Perform regression analysis on the residuals if required
# Example using OLS regression (you would need additional independent variables for a meaningful model)
import statsmodels.api as sm

# Assuming you have another predictor variable called 'predictor'
# train['predictor'] = ... # Add your predictor data here

# Add a constant to the predictor variable
X = sm.add_constant(train['predictor'])

# Build the OLS model (ordinary least squares)
ols_model = sm.OLS(train['SOLD_PRICE'], X)

# Fit the model
ols_results = ols_model.fit()

# Print the summary of the regression
print(ols_results.summary())

# Predicted vs Actual values plot
predictions = ols_results.predict(X)
plt.figure(figsize=(12, 6))
plt.scatter(train['SOLD_PRICE'], predictions, alpha=0.5)
plt.title('Predicted vs Actual Values', fontsize=16)
plt.xlabel('Actual Price', fontsize=14)
plt.ylabel('Predicted Price', fontsize=14)
plt.plot([train['SOLD_PRICE'].min(), train['SOLD_PRICE'].max()], [train['SOLD_PRICE'].min(), train['SOLD_PRICE'].max()], 'r--')
plt.show()


# In[ ]:




