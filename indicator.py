import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


# Function to create color gradients
def cGreen(g):
    if g > 9:
        return "#006400"
    elif g > 8:
        return "#1A741A"
    elif g > 7:
        return "#338333"
    elif g > 6:
        return "#4D934D"
    elif g > 5:
        return "#66A266"
    elif g > 4:
        return "#80B280"
    elif g > 3:
        return "#99C199"
    elif g > 2:
        return "#B3D1B3"
    elif g > 1:
        return "#CCE0CC"
    else:
        return "#E6F0E6"

def cRed(g):
    if g > 9:
        return "#E00000"
    elif g > 8:
        return "#E31A1A"
    elif g > 7:
        return "#E63333"
    elif g > 6:
        return "#E94D4D"
    elif g > 5:
        return "#EC6666"
    elif g > 4:
        return "#F08080"
    elif g > 3:
        return "#F39999"
    elif g > 2:
        return "#F6B3B3"
    elif g > 1:
        return "#F9CCCC"
    else:
        return "#FCE6E6"


def highest_bar_prev(df, n, col_name='high'):
    high_ls = df[col_name].iloc[-n-1:].tolist()
    try:
        _high_df = pd.DataFrame({col_name: high_ls[:-1]})
        idx_max = _high_df[::-1].idxmax()
        idx_max = idx_max+1
        return _high_df.loc[idx_max][col_name].values[0]
    except:
        return high_ls[-1]
    
def lowest_bar_prev(df, n, col_name='low'):
    low_ls = df[col_name].iloc[-n-1:].tolist()
    try:
        _low_df = pd.DataFrame({col_name: low_ls[:-1]})
        idx_min = _low_df[::-1].idxmin()
        idx_min = idx_min+1
        return _low_df.loc[idx_min][col_name].values[0]
    except:
        return low_ls[-1]

def apply_highest_bar_prev(df, n, col_name='high'):
    results = []
    for i in range(len(df)):
        subset = df.iloc[:i + 1]
        result = highest_bar_prev(subset, n, col_name)
        results.append(result)
    return results

def apply_lowest_bar_prev(df, n, col_name='low'):
    results = []
    for i in range(len(df)):
        subset = df.iloc[:i + 1]
        result = lowest_bar_prev(subset, n, col_name)
        results.append(result)
    return results

def calc_high_low(df, min_lookback, middle_lookback, max_lookback):
    df['high1'] = apply_highest_bar_prev(df, min_lookback, col_name='high')
    df['high2'] = apply_highest_bar_prev(df, middle_lookback, col_name='high')
    df['high3'] = apply_highest_bar_prev(df, max_lookback, col_name='high')
    df['low1'] = apply_lowest_bar_prev(df, min_lookback, col_name='low')
    df['low2'] = apply_lowest_bar_prev(df, middle_lookback, col_name='low')
    df['low3'] = apply_lowest_bar_prev(df, max_lookback, col_name='low')
    
    return df

def mean(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)

def variance(x):
    return np.cumsum((x - mean(x))**2) / np.arange(1, len(x) + 1)

def std_dev(x):
    return np.sqrt(variance(x))

def z_norm(x):
    return (x - mean(x)) / std_dev(x)

# Function to check if a value is close to zero
def is_zero(val, tol=0.0015):
    return abs(val) <= tol

# Main function to compute buy and sell signals
def leman_trend_indicator(df, min_lookback=13, middle_lookback=21, max_lookback=34, tol=0.0015):
    # Calculate high and low values over different periods
    df = calc_high_low(df, min_lookback, middle_lookback, max_lookback)
    
    # Calculate buy and sell signals
    df['buy'] = (df['low1'] - df['low']) + (df['low2'] - df['low']) + (df['low3'] - df['low'])
    df['sell'] = (df['high'] - df['high1']) + (df['high'] - df['high2']) + (df['high'] - df['high3'])
    
    # Check for zero difference between buy and sell signals (neutral zone)
    df['zero'] = df['buy'] - df['sell']
    df['zero'] = df['zero'].apply(lambda x: is_zero(x, tol))
    
    # Apply Z-Score normalization
    df['znorm_buy'] = z_norm(df['buy'])
    df['znorm_sell'] = z_norm(df['sell'])

    return df



if __name__ == "__main__":
	
	# Sample data (replace this with the appropriate ohlc data)
	# Needs high, low, columns (lowercase)
	import yfinance as yf
	stock_df = yf.download("AAPL", interval='1d')

	#Converting column names to lowercase
	stock_df.rename(columns=str.lower, inplace=True)

	# Apply LeMan Trend Indicator
	res_df = leman_trend_indicator(stock_df)

	#Update this to plot more of the data
	plot_df = res_df.tail(100)

	# Plotting results
	plt.figure(figsize=(10, 6))
	plt.plot(plot_df.index, plot_df['znorm_buy'], label='Normalized Buy Signal', color='green')
	plt.plot(plot_df.index, plot_df['znorm_sell'], label='Normalized Sell Signal', color='red')

	# Plot neutral zone
	plt.plot(plot_df.index, plot_df['zero'] * 0, label='Neutral Zone', color='gray', linestyle='--')

	# Highlight buy/sell zones
	plt.scatter(plot_df.index[plot_df['zero']], plot_df['znorm_buy'][plot_df['zero']], color='yellow', label='Neutral Signal', zorder=5)

	plt.title('LeMan Trend Indicator')
	plt.legend(loc='best')
	plt.grid(True)
	plt.show()