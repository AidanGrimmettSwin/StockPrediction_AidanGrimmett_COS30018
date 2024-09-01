import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as fplt

def plot_boxplot(data, window_size=5, title="Boxplot Chart"):
    
    # Function plots a boxplot chart for the given stock market data over a moving window.

    # Parameters:
    # - data (pd.DataFrame): dataframe containing the stock market data. must contain the 'Close' and 'Date' features/columns
    # - window_size (int): The size of the moving window (in days) over which to calculate the boxplot statistics. Default is 5 days
    # - title (str): chart title

    # the function calculates the rolling window statistics and generates a boxplot for each window.
    
    # Convert Series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Check if 'Date' is a column or if the index is already datetime
    if 'Date' in data.columns:
        data = data.set_index(pd.DatetimeIndex(data['Date']))
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex.")

    # Rolling window on the 'Close' prices to calculate statistics for each window
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must have a 'Close' column for boxplot calculation.")
    
    # Manually collect boxplot data over rolling windows
    boxplot_data = []
    for i in range(len(data) - window_size + 1):
        window_data = data['Close'].iloc[i:i + window_size]
        if len(window_data.dropna()) == window_size:  # Ensure the window is fully populated
            boxplot_data.append(window_data.values)
    
    # plotting the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, patch_artist=True, showfliers=False)
    plt.title(title)
    plt.xlabel("Date Rolling Window")
    plt.ylabel("Closing Price")
    plt.grid(True)
    plt.show()

def resample_data(data, n):
    # Resamples the data to aggregate every `n` trading days into one

    # Parameters:
    # - data (pd.DataFrame): Original stock market data with DateTimeIndex
    # - n (int): Number of trading days to combine into one candlestick
    
    # Resample the data to create OHLC for every `n` days
    resampled_data = data.resample(f'{n}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    return resampled_data

def plot_candlestick(data, title='Candlestick Plot', n=30):
    # Plots a candlestick chart for the given stock market data

    # inputs:
    # - data (pd.DataFrame): DataFrame containing the stock market data
    # - title (str): Title of the chart. Default is 'Candlestick Plot'
    # - n (int): Number of trading days to combine into one candlestick. Default is 1

    # The function will plot a candlestick chart where each candle represents `n` trading days
    
    # Check if 'Date' is in columns and set it as index
    if 'Date' in data.columns:
        data = data.set_index(pd.DatetimeIndex(data['Date']))

    # Ensure data is sorted by date
    data = data.sort_index()

    # Resample data if n > 1
    if n > 1:
        data = resample_data(data, n)

    # Plotting the candlestick chart using mplfinance
    fplt.plot(
        data, # stock data
        type='candle',  # Specify that we want a candlestick chart
        title=title,    # Title of the plot
        ylabel='Price (Normalised $)',  # Label for the y-axis
        figsize=(10, 6)  # Size of the figure
    )