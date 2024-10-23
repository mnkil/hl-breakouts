# bomy library
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(f'{formatted_date_time}: bomy package import done...')

def test():
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'{formatted_date_time}: bomy package import done...')

def bomy_ts(bomystr=None):
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    if bomystr:
        print(f'{formatted_date_time}: {bomystr}')
    else:
        print(f'{formatted_date_time}:')

def bomy_ts_utc(bomystr=None):
    now = datetime.utcnow()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    if bomystr:
        print(f'{formatted_date_time}: {bomystr}')
    else:
        print(f'{formatted_date_time}:')

def bomy_ts_utc_np(bomystr=None):
    now = datetime.utcnow()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    if bomystr:
        return f'{formatted_date_time}: {bomystr}'
    else:
        return f'{formatted_date_time}:'

def ret_binance_spot(symbol, interval, start_date=None, end_date=None):
    # Binance API key and secret
    with open("discord.yaml", "r") as file:
        data = yaml.safe_load(file)
        api_key = data.get("binance_api_key")[0]
        api_secret = data.get("binance_api_secret")[0]

    # Create a Binance client
    client = Client(api_key, api_secret)
    # Convert start and end dates to milliseconds
    if start_date:
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    else:
        start_ts = int(pd.to_datetime('2016-01-01').timestamp() * 1000)
    if end_date != None:
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
    else:
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        yesterday_date_str = yesterday.strftime('%Y-%m-%d')
        end_date = yesterday_date_str
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

    if interval == '1D':
        interval = Client.KLINE_INTERVAL_1DAY
    if interval == '1H':
        interval = Client.KLINE_INTERVAL_1HOUR

    # klines = client.futures_historical_klines(symbol, interval, start_ts, end_ts)
    klines = client.get_historical_klines(symbol, interval, start_ts, end_ts)

    cols = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']
    df = pd.DataFrame(klines, columns=cols)

    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    # df.set_index('Open Time', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    df = bomypandas(df)
    return df

    # df = ret_binance_spot('BTCUSDT', Client.KLINE_INTERVAL_1DAY, '2017-01-01')
    # df.reset_index(inplace=True)


# yFinance data retrieve
def yfd(ticker, start=None, end=None):
    # Define the ticker symbol
    ticker_symbol = ticker
    # ticker_symbol = "SPY"
    # Fetch the data
    yf_data = yf.Ticker(ticker_symbol)
    if start:
        start = start
    else:
        start ="2023-01-01"
    if end:
        yf_end_date = end
    else:
        now = datetime.now()
        yf_end_date = now.strftime("%Y-%m-%d")
    # Get historical daily data (default is from start till today)
    hist_data = yf_data.history(start=start, end=yf_end_date)
    hist_data = bomypandas(hist_data)
    # Extract the desired columns
    # ohlc_volume_data = hist_data[['Open', 'High', 'Low', 'Close', 'Volume']]

    return hist_data
    # return ohlc_volume_data


def bomy_risk(df, crypto=None):
    import numpy as np

    if crypto:
        ls = [7, 30, 90, 180, 360, 720]
    else:
        ls = [5, 20, 62, 124, 256, 512]

    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['squared_returns'] = df['returns'] ** 2
    df['sstd-w'] = (df['squared_returns'].rolling(window=ls[0]).sum() / 7) ** 0.5 * 365.25 ** 0.5

    df['std-w'] = df['returns'].rolling(window=ls[0]).std() * 365.25 ** 0.5
    df['std-m'] = df['returns'].rolling(window=ls[1]).std() * 365.25 ** 0.5
    df['std-3m'] = df['returns'].rolling(window=ls[2]).std() * 365.25 ** 0.5
    df['std-6m'] = df['returns'].rolling(window=ls[3]).std() * 365.25 ** 0.5
    df['std-1y'] = df['returns'].rolling(window=ls[4]).std() * 365.25 ** 0.5
    df['std-2y'] = df['returns'].rolling(window=ls[5]).std() * 365.25 ** 0.5
    df['std-ewm32'] = df['returns'].ewm(span=32).std() * 365.25 ** 0.5
    df['std-blnd'] = 0.3 * df['std-2y'] + 0.7 * df['std-ewm32']

    return df


def bomy_pt_risk(df, title=None):
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd
    dfg = df
    if title:
        chart_title = title
    else:
        chart_title = ''
    # start_date = '2023-09-01'
    # dfg = dfg[dfg['Open Time'] >= pd.to_datetime(start_date)]

    # Plotting ATR, ExpATR, and their difference using Plotly
    fig = go.Figure()

    # Add ATR trace
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['std7'], mode='lines', name='std7'))
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['std30'], mode='lines', name='std30'))
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['std90'], mode='lines', name='std90'))
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['std360'], mode='lines', name='std360'))
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['stdewm32'], mode='lines', name='stdewm32'))
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['stdbld'], mode='lines', name='stdbld'))
    # fig.add_trace(go.Scatter(x=np.array(dfg['Open Time']), y=dfg['sstd7'], mode='lines', name='sstd7'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['std-w'], mode='lines', name='std-w'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['std-m'], mode='lines', name='std-m'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['std-3m'], mode='lines', name='std-3m'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['std-1y'], mode='lines', name='std-1y'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['std-ewm32'], mode='lines', name='std-ewm32'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['std-blnd'], mode='lines', name='std-blnd'))
    fig.add_trace(go.Scatter(x=np.array(dfg['ts_date']), y=dfg['sstd-w'], mode='lines', name='sstd-w'))


    # Update layout
    fig.update_layout(
        title=chart_title,
        # xaxis_title='Open Time',
        yaxis_title='%'
        # legend_title='Indicators'
    )

    # Show plot
    fig.show()

def bomypandas(df):
    # What columns should a bomy pandas have:
    # Close
    # ts_date (date)
    bomy_ts('bomyPandas init...')
    bdf = df.reset_index()
    bomy_ts(f' {type(bdf)}')
    bomy_ts('bomyPandas columns -->')
    print('---------------------------------------')
    if 'Date' in bdf.columns:
        bdf.rename(columns={'Date': 'ts_date'}, inplace=True)
    if 'Close Time' in bdf.columns:
        bdf['ts_date'] = bdf['Close Time']
    for column_name, dtype in bdf.dtypes.items():
        print(f"{column_name}: {dtype}")
    print('---------------------------------------')
    bomy_ts('bomyPandas done...')
    return bdf


def calcDonchianChannels(data: pd.DataFrame, period: int):
    # Standard period is 20
    bomy_ts('Calculating Donchian Channels...')
    data["upperDon"] = data["High"].rolling(period).max()
    data["lowerDon"] = data["Low"].rolling(period).min()
    data["midDon"] = (data["upperDon"] + data["lowerDon"]) / 2
    return data

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def ptDonchianChannels(data: pd.DataFrame, Asset=None):
    data.index = data.ts_date
    plt.figure(figsize=(12, 8))
    plt.plot(data["Close"], label="Close")
    plt.plot(data["upperDon"], label="Upper", c=colors[1])
    plt.plot(data["lowerDon"], label="Lower", c=colors[4])
    plt.plot(data["midDon"], label="Mid", c=colors[2], linestyle=":")
    plt.fill_between(data.index, data["upperDon"], data["lowerDon"], alpha=0.3,
                     color=colors[1])
    if Asset:
        ticker = Asset
        plt.title(f"Donchian Channels for {ticker}")
    else:
        print(f'Donchian Channels')
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.title(f"Donchian Channels for {ticker}")
    ax = plt.gca()  # Get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

def midDonCrossOver(data: pd.DataFrame, period: int=20, shorts: bool=True):
    data = calcDonchianChannels(data, period)

    data["position"] = np.nan
    data["position"] = np.where(data["Close"]>data["midDon"], 1,
                              data["position"])
    if shorts:
        data["position"] = np.where(data["Close"]<data["midDon"], -1, data["position"])
    else:
        data["position"] = np.where(data["Close"]<data["midDon"], 0, data["position"])
    data["position"] = data["position"].ffill().fillna(0)

    return calcReturns(data)

def ptmidDonCrossOver(midDon: midDonCrossOver):
    plt.figure(figsize=(12, 4))
    plt.plot(midDon["strat_cum_returns"] * 100, label="Mid Don X-Over")
    plt.plot(midDon["cum_returns"] * 100, label="Buy and Hold")
    plt.title("Cumulative Returns for Mid Donchian Cross-Over Strategy")
    plt.xlabel("Date")
    plt.ylabel("Returns (%)")
    ax = plt.gca()  # Get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()

    stats = pd.DataFrame(getStratStats(midDon["log_returns"]),
                         index=["Buy and Hold"])
    stats = pd.concat([stats,
                       pd.DataFrame(getStratStats(midDon["strat_log_returns"]),
                                   index=["MidDon X-Over"])])
    print(stats)


def donChannelBreakout(data, period=20, shorts=True):
    data = calcDonchianChannels(data, period)

    data["position"] = np.nan
    data["position"] = np.where(data["Close"] > data["upperDon"].shift(1), 1,
                                data["position"])
    if shorts:
        data["position"] = np.where(
            data["Close"] < data["lowerDon"].shift(1), -1, data["position"])
    else:
        data["position"] = np.where(
            data["Close"] < data["lowerDon"].shift(1), 0, data["position"])

    data["position"] = data["position"].ffill().fillna(0)

    return calcReturns(data)

def ptdonChannelBreakout(breakout):
    breakout = breakout

    plt.figure(figsize=(12, 4))
    plt.plot(breakout["strat_cum_returns"] * 100, label="Donchian Breakout")
    plt.plot(breakout["cum_returns"] * 100, label="Buy and Hold")
    plt.title("Cumulative Returns for Donchian Breakout Strategy")
    plt.xlabel("Date")
    plt.ylabel("Returns (%)")
    ax = plt.gca()  # Get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()

    stats = pd.concat([pd.DataFrame(getStratStats(breakout["strat_log_returns"]), index=["Donchian Breakout"])])
    print(stats)


def donReversal(data, period=20, shorts=True):
    data = calcDonchianChannels(data, period)

    data["position"] = np.nan
    data["position"] = np.where(data["Close"] < data["lowerDon"].shift(1),
                                1, data["position"])

    short_val = -1 if shorts else 0
    data["position"] = np.where(data["Close"] > data["lowerDon"].shift(1),
                                short_val, data["position"])

    data["position"] = data["position"].ffill().fillna(0)

    return calcReturns(data)

def ptDonReversal(donRev):
    donRev = donRev

    plt.figure(figsize=(12, 4))
    plt.plot(donRev["strat_cum_returns"] * 100, label="Donchian Reversal")
    plt.plot(donRev["cum_returns"] * 100, label="Buy and Hold")
    plt.title("Cumulative Returns for Donchian Reversal Strategy")
    plt.xlabel("Date")
    plt.ylabel("Returns (%)")
    plt.xticks(rotation=45)
    plt.legend()

    plt.show()

    stats = pd.concat([pd.DataFrame(getStratStats(donRev["strat_log_returns"]), index=["Donchian Reversal (20-Day)"])])
    print(stats)

# A few helper functions
def calcReturns(df):
    # Helper function to avoid repeating too much code
    df['returns'] = df['Close'] / df['Close'].shift(1)
    df['log_returns'] = np.log(df['returns'])
    df['strat_returns'] = df['position'].shift(1) * df['returns']
    df['strat_log_returns'] = df['position'].shift(1) * \
                              df['log_returns']
    df['cum_returns'] = np.exp(df['log_returns'].cumsum()) - 1
    df['strat_cum_returns'] = np.exp(
        df['strat_log_returns'].cumsum()) - 1
    df['peak'] = df['cum_returns'].cummax()
    df['strat_peak'] = df['strat_cum_returns'].cummax()
    return df


def getStratStats(log_returns: pd.Series, risk_free_rate: float = 0.02):
    stats = {}  # Total Returns
    stats['tot_returns'] = np.exp(log_returns.sum()) - 1

    # Mean Annual Returns
    stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1

    # Annual Volatility
    stats['annual_volatility'] = log_returns.std() * np.sqrt(252)

    # Sortino Ratio
    annualized_downside = log_returns.loc[log_returns < 0].std() * \
                          np.sqrt(252)
    stats['sortino_ratio'] = (stats['annual_returns'] - \
                              risk_free_rate) / annualized_downside

    # Sharpe Ratio
    stats['sharpe_ratio'] = (stats['annual_returns'] - \
                             risk_free_rate) / stats['annual_volatility']

    # Max Drawdown
    cum_returns = log_returns.cumsum() - 1
    peak = cum_returns.cummax()
    drawdown = peak - cum_returns
    max_idx = drawdown.argmax()
    stats['max_drawdown'] = 1 - np.exp(cum_returns[max_idx]) \
                            / np.exp(peak[max_idx])

    # Max Drawdown Duration
    # strat_dd = drawdown[drawdown == 0]
    # strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    # strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    # strat_dd_days = np.hstack([strat_dd_days,
    #                           (drawdown.index[-1] - strat_dd.index[-1]).days])
    # stats['max_drawdown_duration'] = strat_dd_days.max()
    return {k: np.round(v, 4) if type(v) == np.float_ else v
            for k, v in stats.items()}
