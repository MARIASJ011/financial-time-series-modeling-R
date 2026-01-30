# Apple Stock Time Series Analysis

This assignment focuses on applying **time series analysis techniques** to historical **Apple (AAPL) stock price data**. The goal was to understand price behavior over time, transform raw price data into returns, and analyze volatility patterns using standard financial time series methods.

## Objective
- Analyze historical stock price data for Apple Inc.
- Transform prices into **log returns**
- Examine time-dependent patterns and volatility
- Apply foundational time series concepts used in financial analytics

## Dataset
- Historical Apple stock price data
- Variables include date and closing price
- Data was cleaned and ordered chronologically before analysis

## Methodology

### 1) Data Preparation
- Converted date fields into proper time formats
- Sorted observations by time
- Cleaned price fields and handled missing values

### 2) Return Calculation
- Computed **log returns** from closing prices
- Removed missing values introduced by differencing

### 3) Exploratory Time Series Analysis
- Visualized price movements and return series
- Examined distributional and volatility patterns
- Identified potential time-varying variance in returns

### 4) Volatility Analysis (Introductory)
- Tested for **ARCH effects** to assess volatility clustering
- Laid the groundwork for more advanced volatility models in later assignments

## Key Insights
- Apple stock returns exhibit **volatility clustering**, a common feature in financial time series
- Log returns provide a more stable basis for modeling than raw prices
- Preliminary tests suggest time-varying variance rather than constant volatility

## Tools & Technologies
- **R / Python** (depending on implementation)
- Time series and statistical libraries
- Data visualization tools for trend and volatility inspection
