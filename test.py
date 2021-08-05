# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import time
import ccxt

api_key = ''
api_secret = ''

"""""
def rsi_tradingview(ohlc: pd.DataFrame, period: int = 14, round_rsi: bool = True):
    delta = ohlc["close"].diff()

    up = delta.copy()
    up[up < 0] = 0
    up = pd.Series.ewm(up, alpha=1/period).mean()

    down = delta.copy()
    down[down > 0] = 0
    down *= -1
    down = pd.Series.ewm(down, alpha=1/period).mean()

    rsi = np.where(up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down))))

    return np.round(rsi, 2) if round_rsi else rsi



print(ccxt.exchanges) # print a list of all available exchange classes
# Create instance for your exchange, here binance
binance = ccxt.binance()
# Quick test to verify data access
# Get the last 2 hours candelsticks from the pair 'BTC/USDT'
pair = 'BTC/USDT' 
binance.fetch_ohlcv(pair, limit=2)

current_milli_time = lambda x: int(round((time.time()- 3600*x) * 1000))
# install pandas with pip install pandas, perfect library for manipulate our dataset
import pandas as pd
symbol = 'BTC/USDT'
print(symbol)
ohlcv_dataframe = pd.DataFrame()
for hours in range(4320,0,-600): # 6 month is around 24hours * 30days * 1 = 720
    if binance.has['fetchOHLCV']:
        time.sleep (binance.rateLimit / 1000) # time.sleep wants seconds
        # the limit from binance is 1000 timesteps
        ohlcv = binance.fetch_ohlcv(symbol, '1h', since=current_milli_time(hours),
                                    limit=1000)
        ohlcv_dataframe = ohlcv_dataframe.append(pd.DataFrame(ohlcv))
        print(hours)
# We are changing the name of the columns, important to use trading indicators later on
ohlcv_dataframe['date']   = ohlcv_dataframe[0]
ohlcv_dataframe['open']   = ohlcv_dataframe[1]
ohlcv_dataframe['high']   = ohlcv_dataframe[2]
ohlcv_dataframe['low']    = ohlcv_dataframe[3]
ohlcv_dataframe['close']  = ohlcv_dataframe[4]
ohlcv_dataframe['volume'] = ohlcv_dataframe[5]
ohlcv_dataframe = ohlcv_dataframe.set_index('date')
# Change the timstamp to date in UTC
ohlcv_dataframe = ohlcv_dataframe.set_index(
    pd.to_datetime(ohlcv_dataframe.index, unit='ms').tz_localize('UTC'))
ohlcv_dataframe.drop([0,1,2,3,4,5], axis=1, inplace=True)
# Create CSV file from our panda dataFrame
ohlcv_dataframe.to_csv('data_since6months_freq1h'+symbol.split('/')[0]+'.csv')

# Read data from csv file to a dataframe
symbol = 'BTC/USDT'
data = pd.read_csv('data_since6months_freq1h'+symbol.split('/')[0]+'.csv', 
                   index_col="date")
# Extra precaution to ensure correct data: remove potential duplicate 
data.index = pd.DatetimeIndex(data.index)
data = data[~data.index.duplicated(keep='first')]
# Reindex date approriately to easily spot missing data with NaN value
data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1h'))

rsi = rsi_tradingview(data)
print("RSI: ")
for i in range(len(rsi)):
    print(f"Date={data.index[i]}\tRSI={rsi[i]}")
    
"""

import re
from stockstats import StockDataFrame as Sdf
# configure exchange
#########Hello, fill the api key and secret key for binance api, don't share them with anyone!!!
exchange = getattr(ccxt, 'binance')({
  'apiKey': '',
  'secret': '',
  'timeout': 10000,
  'enableRateLimit': True
})

# load markets and all coin_pairs
exchange.load_markets()
coin_pairs = exchange.symbols
# list of coin pairs which are active and use BTC as base coin
valid_coin_pairs = []
# load only coin_pairs which match regex and are active
regex = '^.*/BTC'

for coin_pair in coin_pairs:
  if re.match(regex, coin_pair) and exchange.markets[coin_pair]['active']:
    valid_coin_pairs.append(coin_pair)

def get_historical_data(coin_pair, timeframe):
    """Get Historical data (ohlcv) from a coin_pair
    """
    # optional: exchange.fetch_ohlcv(coin_pair, '1h', since)
    ###########Hello, the last parameter is for historical data, you can increaset it.
    data = exchange.fetch_ohlcv(coin_pair, timeframe, None, 2000)
    # update timestamp to human readable timestamp
    data = [[exchange.iso8601(candle[0])] + candle[1:] for candle in data]
    header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(data, columns=header)
    return df

def create_stock(historical_data):
    """Create StockData from historical data 
    """
    stock  = Sdf.retype(historical_data)
    return stock


def get_rsi_macd():
    #for coin_pair in valid_coin_pairs:
    # respect rate limit
    #time.sleep (exchange.rateLimit / 1000)
    data = get_historical_data('BTC/USDT', '1m')
    stock_data = create_stock(data)
    # stock_data contains historical data of ETH/BTC with a period of 1 hhour
    # the volume is calculated by using historical data 
    # run this as close to each 1h interval (e.g. 7.59h or 9.59m)
    last_items = stock_data.tail(24)
    #print(last_items)
    day_volume_self_calculated = last_items['volume'].sum()
    #print( day_volume_self_calculated)

    # better way to do it
    ticker = exchange.fetch_ticker(coin_pair)
    day_volume = ticker["baseVolume"]
    #print(day_volume)
    
    # quote volume (expressed in BTC value)
    day_volume_in_btc = ticker["quoteVolume"]
    #print( day_volume_in_btc)
    
    # Calculate RSI                       
    stock_data['rsi_14']                     
    #print( stock_data)
                           
    # Get most recent RSI value of our data frame
    # In our case this represents the RSI of the last 1h                  
    last_rsi = stock_data['rsi_14'].iloc[-1]                       
    print('rsi', last_rsi)
    
    # Calculate macd
    stock_data['macd']
    #print(stock_data)
    
    last_macd = stock_data['macd'].iloc[-1]
    print('macd', last_macd)
    

if __name__ == '__main__':
    index = 0
    ########hello, this is the time to show, 300s default, you can change this as you want.
    while (index<300):
        time.sleep(1)
        print("----------------------")
        get_rsi_macd()
        index += 1
