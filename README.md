# Time Series Modeling for Apple Stocks
#-------------------------------# 
# Step 1: Load and inspect data 
#-------------------------------# 
install.packages("dplyr") 
install.packages("ggplot2") 
install.packages("sf") 
install.packages("tidyverse") 
install.packages("lubridate") 
install.packages("zoo") 
install.packages("tsibble") 
install.packages("fable") 
install.packages("feasts") 
install.packages("tidyr") 
install.packages("forecast") 
install.packages("keras") 
install.packages("tensorflow") 
install.packages("randomForest") 
install.packages("rpart") 
19 
library(dplyr) 
library(ggplot2) 
library(sf) 
library(tidyverse) 
library(lubridate) 
library(zoo) 
library(tsibble) 
library(fable) 
library(feasts) 
library(tidyr) 
library(forecast) 
library(keras) 
library(tensorflow) 
library(randomForest) 
library(rpart) 
# Read data 
apple <- read.csv("D:\\Data\\AAPL Historical Data (2).csv") 
# Convert columns 
apple$Date <- dmy(apple$Date) 
apple$Price <- as.numeric(gsub(",", "", apple$Price)) 
apple$Vol. <- as.numeric(gsub("M", "e6", gsub("K", "e3", gsub(",", "", apple$Vol.)))) 
20 
apple$Change... <- as.numeric(gsub("%", "", apple$Change..)) / 100 
# Sort by date 
apple <- apple[order(apple$Date), ] 
# Interpolate missing values 
apple <- apple %>% 
mutate(across(where(is.numeric), ~ zoo::na.approx(., na.rm = FALSE))) 
# Create lag feature before splitting 
apple <- apple %>% 
arrange(Date) %>% 
mutate(Lag1 = lag(Price, 1)) 
# Plot Price over Time 
ggplot(apple, aes(x = Date, y = Price)) + 
geom_line(color = "darkgreen") + 
labs(title = "Apple Stock Price Over Time", x = "Date", y = "Price (USD)") + 
theme_minimal() 
# Split into Train/Test sets 
n <- nrow(apple) 
train_index <- 1:round(0.8 * n) 
apple_train <- apple[train_index, ] 
21 
apple_test <- apple[-train_index, ] 
#-------------------------------# 
# Step 2: Monthly Aggregation & STL Decomposition 
#-------------------------------# 
# Convert to tsibble 
apple_ts <- apple %>% 
select(Date, Price) %>% 
as_tsibble(index = Date) 
# Aggregate to monthly 
apple_monthly <- apple_ts %>% 
mutate(Month = floor_date(Date, "month")) %>% 
group_by(Month) %>% 
summarise(Price = mean(Price, na.rm = TRUE)) %>% 
ungroup() %>% 
as_tsibble(index = Month, key = NULL) 
# Fill missing months and values 
apple_monthly <- apple_monthly %>% 
fill_gaps() %>% 
fill(Price, .direction = "downup") 
# STL Decomposition 
22 
apple_monthly %>% 
model( 
STL = STL(Price ~ trend(window = 13) + season(window = "periodic")) 
) %>% 
components() %>% 
autoplot() 
#-------------------------------# 
# Step 3: Univariate Forecasting 
#-------------------------------# 
# Holt-Winters Forecasting 
hw_model <- HoltWinters(ts(apple_monthly$Price, frequency = 12)) 
plot(hw_model) 
forecast_hw <- forecast(hw_model, h = 12) 
plot(forecast_hw) 
# ARIMA Daily 
auto_arima_daily <- auto.arima(apple$Price) 
checkresiduals(auto_arima_daily) 
forecast(auto_arima_daily, h = 90) %>% autoplot() 
# SARIMA Daily 
sarima_model <- auto.arima(apple$Price, seasonal = TRUE) 
checkresiduals(sarima_model) 
forecast(sarima_model, h = 90) %>% autoplot() 
23 
# ARIMA Monthly 
monthly_ts <- ts(apple_monthly$Price, frequency = 12) 
arima_monthly <- auto.arima(monthly_ts) 
forecast(arima_monthly, h = 3) %>% autoplot() 
#-------------------------------# 
# Step 4: Multivariate Forecasting 
#-------------------------------# 
# Random Forest 
rf_model <- randomForest(Price ~ Lag1, data = na.omit(apple_train)) 
pred_rf <- predict(rf_model, newdata = na.omit(apple_test)) 
# Decision Tree 
dt_model <- rpart(Price ~ Lag1, data = na.omit(apple_train)) 
pred_dt <- predict(dt_model, newdata = na.omit(apple_test)) 
#-------------------------------# 
# Step 4.1: Plot Predictions vs Actuals 
#-------------------------------# 
# Extract matching actual prices for test set 
actual <- na.omit(apple_test)$Price 
dates <- na.omit(apple_test)$Date 
24 
# Create a data frame for plotting 
comparison_df <- data.frame( 
Date = dates, 
Actual = actual, 
Random_Forest = pred_rf, 
Decision_Tree = pred_dt 
) 
# Melt the dataframe for ggplot 
library(reshape2) 
comparison_long <- melt(comparison_df, id.vars = "Date", variable.name = "Model", 
value.name = "Price") 
# Plot 
library(ggplot2) 
ggplot(comparison_long, aes(x = Date, y = Price, color = Model)) + 
geom_line(size = 1) + 
labs(title = "ML Model Forecasts vs Actual Prices", 
x = "Date", 
y = "Stock Price (USD)", 
color = "Legend") + 
theme_minimal() + 
scale_color_manual(values = c("Actual" = "black", "Random_Forest" = "blue", 
"Decision_Tree" = "red")) 
25 
Python: 
# ============================== # 
# 1. Libraries and Data Loading # 
# ============================== # 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from statsmodels.tsa.arima.model import ARIMA 
import statsmodels.api as sm 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, r2_score 
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Input 
# Load data 
df = pd.read_csv("//content//AAPL Historical Data (4).csv")  # Change to your Apple CSV file 
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)  # Adjust format if needed 
df.set_index('Date', inplace=True) 
df.sort_index(inplace=True) 
26 
# Fill missing prices 
df['Price'].interpolate(method='linear', inplace=True) 
# ============================== # 
# 2. Time Series Visualization  # 
# ============================== # 
plt.figure(figsize=(12, 6)) 
df['Price'].plot(title="Apple Monthly Close Price") 
plt.xlabel("Date") 
plt.ylabel("Price ($)") 
plt.grid(True) 
plt.show() 
# ================================ # 
# 3. Decomposition (Seasonal-Trend) # 
# ================================ # 
# Additive 
add = seasonal_decompose(df['Price'], model='additive', period=12) 
add.plot() 
plt.suptitle("Additive Decomposition", y=1.02) 
plt.show() 
# Multiplicative 
27 
mult = seasonal_decompose(df['Price'], model='multiplicative', period=12) 
mult.plot() 
plt.suptitle("Multiplicative Decomposition", y=1.02) 
plt.show() 
# ========================== # 
# 4. Holt-Winters Forecasting # 
# ========================== # 
hw_model = ExponentialSmoothing(df['Price'], seasonal='add', seasonal_periods=12).fit() 
hw_forecast = hw_model.forecast(12) 
plt.figure(figsize=(12, 6)) 
df['Price'].plot(label='Actual') 
hw_forecast.plot(label='Holt-Winters Forecast', color='red') 
plt.title("Holt-Winters 12-Month Forecast (Apple)") 
plt.xlabel("Date") 
plt.ylabel("Price ($)") 
plt.legend() 
plt.grid(True) 
plt.show() 
# ============= # 
# 5. ARIMA      # 
# ============= # 
28 
arima_model = ARIMA(df['Price'], order=(1, 1, 1)).fit() 
arima_model.plot_diagnostics(figsize=(10, 6)) 
plt.tight_layout() 
plt.show() 
arima_forecast = arima_model.forecast(steps=3) 
print("ARIMA 3-Month Forecast:\n", arima_forecast) 
# ============= # 
# 6. SARIMA     # 
# ============= # 
sarima_model = sm.tsa.SARIMAX(df['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit() 
sarima_model.plot_diagnostics(figsize=(10, 6)) 
plt.tight_layout() 
plt.show() 
sarima_forecast = sarima_model.forecast(steps=3) 
print("SARIMA 3-Month Forecast:\n", sarima_forecast) 
# =================== # 
# 7. LSTM Forecasting # 
# =================== # 
# Scale the Price 
scaler = MinMaxScaler() 
29 
scaled_prices = scaler.fit_transform(df[['Price']]) 
# Create Sequences 
window_size = 12 
X_lstm, y_lstm = [], [] 
for i in range(window_size, len(scaled_prices)): 
X_lstm.append(scaled_prices[i-window_size:i]) 
y_lstm.append(scaled_prices[i]) 
X_lstm = np.array(X_lstm) 
y_lstm = np.array(y_lstm) 
# Train-Test Split 
split_index = int(0.8 * len(X_lstm)) 
X_train, X_test = X_lstm[:split_index], X_lstm[split_index:] 
y_train, y_test = y_lstm[:split_index], y_lstm[split_index:] 
# Build LSTM Model 
model = Sequential() 
model.add(Input(shape=(X_train.shape[1], 1))) 
model.add(LSTM(50, activation='relu')) 
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse') 
# Train the Model 
model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1) 
30 
# Predict 
y_pred_scaled = model.predict(X_test) 
y_pred = scaler.inverse_transform(y_pred_scaled) 
y_test_actual = scaler.inverse_transform(y_test) 
# Plot Results 
plt.figure(figsize=(10, 5)) 
plt.plot(y_test_actual, label='Actual Price') 
plt.plot(y_pred, label='LSTM Forecast', linestyle='--') 
plt.title("Apple Stock Price Forecast using LSTM") 
plt.xlabel("Time (months)") 
plt.ylabel("Price ($)") 
plt.legend() 
plt.grid(True) 
plt.tight_layout() 
plt.show() 
# Evaluation 
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred)) 
r2 = r2_score(y_test_actual, y_pred) 
print(f"RMSE: {rmse:.2f}") 
print(f"R² Score: {r2:.4f}")
