# ============================== #
# 1. Load Required Libraries
# ============================== #
library(tidyverse)
library(lubridate)
library(tseries)
library(rugarch)

# ============================== #
# 2. Load and Clean the Dataset
# ============================== #
data <- read.csv("D:\\Data\\Apple Stock Price History (1).csv")

# Convert 'Date' to Date format
data$Date <- dmy(data$Date)

# Sort by ascending date
data <- data[order(data$Date), ]

# Remove commas and convert 'Price' to numeric
data$Price <- as.numeric(gsub(",", "", data$Price))

# Remove NA values
data <- na.omit(data)

# ============================== #
# 3. Calculate Log Returns
# ============================== #
returns <- diff(log(data$Price))
returns <- na.omit(returns)

# ============================== #
# 4. ARCH Effect Test
# ============================== #
# Use safe lag (minimum of 12 or one-fifth of data length)
safe_lag <- min(12, floor(length(returns) / 5))

if (length(returns) > safe_lag) {
  arch_test <- ArchTest(returns, lags = safe_lag)
  print(arch_test)
} else {
  cat("Not enough data to run ARCH test.\n")
}

# ============================== #
# 5. GARCH(1,1) Model Estimation
# ============================== #
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0)),
  distribution.model = "norm"
)

fit <- ugarchfit(spec = spec, data = returns)
show(fit)

# ============================== #
# 6. Forecast 3-Month Volatility (60 Days)
# ============================== #
forecast <- ugarchforecast(fit, n.ahead = 60)
vol_forecast <- sigma(forecast)

# Plot Forecast
plot(vol_forecast, type = "l", col = "blue", main = "60-Day Volatility Forecast", ylab = "Volatility", xlab = "Days Ahead")
# ============================== #
#      PART B – VAR/VECM        #
# ============================== #


library(vars)
library(urca)
library(readr)
library(dplyr)
library(tseries)

# Load data (assumes Date + 6 commodities)
data <- read_csv("D:\\Data\\Commodity_Prices.csv")

# Convert to time series
data$Date <- as.Date(data$Date)
ts_data <- ts(data[,-1], start = c(2010, 1), frequency = 12)  # Adjust based on your date range

# Check stationarity with ADF test
apply(ts_data, 2, function(x) adf.test(x)$p.value)

# Difference if non-stationary
diff_data <- diff(ts_data)

# VAR model
var_model <- VAR(diff_data, p = 2, type = "const")
summary(var_model)

# Check for cointegration
ca_test <- ca.jo(ts_data, type = "trace", ecdet = "const", K = 2)
summary(ca_test)

# If cointegration exists (e.g., trace stat > critical value)
vecm <- cajorls(ca_test, r = 1)
summary(vecm)

# ============================== #
#       Forecast using VECM     #
# ============================== #

library(forecast)

# Convert VECM to VAR
vec2var_model <- vec2var(ca_test, r = 1)

# Forecast using VECM (20 months ahead)
vecm_forecast <- predict(vec2var_model, n.ahead = 20, ci = 0.95)

# Reconstruct level forecasts
n_ahead <- 20
last_values <- tail(ts_data, 1)
forecast_levels_vecm <- matrix(NA, nrow = n_ahead + 1, ncol = ncol(ts_data))
forecast_levels_vecm[1, ] <- last_values
colnames(forecast_levels_vecm) <- colnames(ts_data)

for (i in 1:n_ahead) {
  for (j in 1:ncol(ts_data)) {
    forecast_levels_vecm[i + 1, j] <- forecast_levels_vecm[i, j] + vecm_forecast$fcst[[j]][i, "fcst"]
  }
}

# ============================== #
#         Plot VECM Forecast     #
# ============================== #
matplot(forecast_levels_vecm, type = "l", lty = 1,
        col = 1:ncol(ts_data), ylab = "Forecasted Price",
        xlab = "Months Ahead", main = "20-Month VECM Forecast")
legend("topleft", legend = colnames(ts_data), col = 1:ncol(ts_data), lty = 1)

# ============================== #
#         Forecast using VAR     #
# ============================== #
var_forecast <- predict(var_model, n.ahead = 20, ci = 0.95)

# Reconstruct level forecasts
forecast_levels_var <- matrix(NA, nrow = n_ahead + 1, ncol = ncol(ts_data))
forecast_levels_var[1, ] <- last_values
colnames(forecast_levels_var) <- colnames(ts_data)

for (i in 1:n_ahead) {
  for (j in 1:ncol(ts_data)) {
    forecast_levels_var[i + 1, j] <- forecast_levels_var[i, j] + var_forecast$fcst[[j]][i, "fcst"]
  }
}

# ============================== #
#        Plot VAR Forecast       #
# ============================== #
matplot(forecast_levels_var, type = "l", lty = 1,
        col = 1:ncol(ts_data), ylab = "Forecasted Price",
        xlab = "Months Ahead", main = "20-Month VAR Forecast")
legend("topleft", legend = colnames(ts_data), col = 1:ncol(ts_data), lty = 1)

# ============================== #
#   Comparison Plot (VECM vs VAR)
# ============================== #
matplot(forecast_levels_vecm, type = "l", lty = 1, col = 1:ncol(ts_data),
        ylab = "Forecasted Prices", xlab = "Months Ahead", main = "VECM vs VAR Forecasts")
matlines(forecast_levels_var, lty = 2, col = 1:ncol(ts_data))
legend("bottomright", legend = c(paste0(colnames(ts_data), " (VECM)"),
                                 paste0(colnames(ts_data), " (VAR)")),
       col = rep(1:ncol(ts_data), 2), lty = rep(1:2, each = ncol(ts_data)))









