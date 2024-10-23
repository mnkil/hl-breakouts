import sqlite3
import yaml
from binance.client import Client
import pandas as pd
from discordwebhook import Discord
import bomy
from datetime import datetime, timedelta
import eth_account
from eth_account.signers.local import LocalAccount
import json
import os
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

with open("discord.yaml", "r") as file:
    data = yaml.safe_load(file)
    discord_url = data.get("discord_url")[0]
    client1 = data.get("client1")[0]
    client2 = data.get("client2")[0]
    discord_log_url = data.get("discord_log_url")[0]
    user_state = data.get("user_state")[0]

# info = Info(constants.TESTNET_API_URL, skip_ws=True)
info = Info(constants.MAINNET_API_URL, skip_ws=True)
user_state = info.user_state(user_state)
print(user_state)

# Set to 0 for Production
TEST_TRADE = 0
TEST_CLOSE = 0

# Read tickers from YAML file
with open("tickers.yaml", "r") as file:
    tickers = yaml.safe_load(file)["tickers"]
    print(f'tickers: {tickers}')

with open("oms.yaml", "r") as file:
    notional_usd = yaml.safe_load(file)["notional_usd"][0]

print('### BREAKOUT SCREENER ###')
print('\n')

# discord webhook
discord = Discord(url=discord_url)
client = Client(client1, client2)
discord_log = Discord(url=discord_log_url)

# SQLite DB historical prices
conn = sqlite3.connect('historical_data.db')
cursor = conn.cursor()

# tickers = ['WIFUSDT']
# Function to calculate Donchian channels

def ctrade(ticker):
    print(print(f'Closing all trades in {ticker}...'))
    address, info, exchange = setup(constants.MAINNET_API_URL, skip_ws=True)

    coin = ticker[:-4]
    if coin == "BONK":
        coin = "kBONK"
    elif coin == "PEPE":
        coin = "kPEPE"
    # coin = "kBONK"
    print(f"We try to Market Close all {coin}.")
    order_result = exchange.market_close(coin)
    try:
        if order_result["status"] == "ok":
            for status in order_result["response"]["data"]["statuses"]:
                try:
                    filled = status["filled"]
                    print(f'Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                    discord.post(content=f'All Trades closed in {ticker}...')
                except KeyError:
                    print(f'Error: {status["error"]}')
                    discord.post(content=f'Failed to close trades in {ticker}...')
    except TypeError as e:
        print(f'no order result therefore error: {e}')


def ntrade(ticker, size):
    address, info, exchange = setup(constants.MAINNET_API_URL, skip_ws=True)

    # coin = "ETH"
    coin = ticker[:-4]
    is_buy = True
    # sz = 1
    sz = size
    print(coin)

    if str(coin) == 'BONK':
        print('BONK adjustment in ntrade()')
        coin = 'kBONK'
        sz = round(sz / 1000, 0)
    elif str(coin) == 'PEPE':
        print('PEPE adjustment in ntrade()')
        coin = 'kPEPE'
        sz = round(sz / 1000, 0)

    print(f"market {'buy' if is_buy else 'sell'} {sz} {coin}.")

    order_result = exchange.market_open(coin, is_buy, sz, None, 0.1)
    if order_result["status"] == "ok":
        for status in order_result["response"]["data"]["statuses"]:
            try:
                filled = status["filled"]
                print(f'Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
            except KeyError:
                print(f'Error: {status["error"]}')
                discord.post(content=f'Trade Execution failed in {ticker}...')
        return filled['oid'], filled['totalSz'], filled['avgPx']


def ems(ticker, size):
    print(f'EMS init {ticker}, {size}...')
    oid, tsize, avgPx = ntrade(ticker, size)
    connt = sqlite3.connect('trades.db')
    trailing = avgPx = 0.9
    data = {
        'timestamp': [datetime.now()],
        'ticker': [ticker],
        'oid': [oid],
        'tsize': [tsize],
        'avgPx': [avgPx],
        'trailing': [trailing]
    }
    dfe = pd.DataFrame(data)
    dfe.to_sql('trades', connt, if_exists='append', index=False)
    connt.close()


# Function to check if there was a breakout in the last 60 minutes
def oms(ticker, current_price):
    print(f"OMS triggered for {ticker}")
    # Connect to the SQLite database
    conn_o = sqlite3.connect('oms.db', uri=True)
    size = notional_usd / current_price
    if size > 10:
        size = round(size, 0)
    elif size > 1:
        size = round(size, 1)
    elif size > 0.1:
        size = round(size, 2)
    else:
        size = round(size, 3)

    dfo = pd.read_sql_query("SELECT * FROM oms", conn_o)
    current_timestamp = datetime.now()
    # Check if there's an entry for the given ticker
    if ticker in dfo['ticker'].values:
        # Check the open_trade value for the ticker
        open_trade_value = dfo.loc[dfo['ticker'] == ticker, 'open_trade'].values[0]
        if open_trade_value == 0:
            discord.post(content=f'No open trade in {ticker}.. -> feed EMS to open new position...')
            # Update the open_trade value to 1 and update the last_update timestamp
            conn_o.execute("UPDATE oms SET open_trade = 1, last_update = ?, txpx = ?, pxhw = ? WHERE ticker = ?",
                           (current_timestamp, current_price, current_price, ticker))
            ems(ticker, size)
            print(f"Updated open_trade value for {ticker} to 1 and last_update timestamp")
        else:
            print(f"Open trade already exists for {ticker}")
            discord.post(content=f'Open trade already exists for {ticker}')
            # Fetch pxhw value from the database for the given ticker
            pxhw = dfo.loc[dfo['ticker'] == ticker, 'pxhw'].values[0]
            print('DEBUG')
            print(pxhw)
            discord.post(content=f'pxhw: {pxhw}')
            # Check if current_price is greater than pxhw
            if current_price > pxhw:
                # Update pxhw with current_price
                conn_o.execute("UPDATE oms SET pxhw = ? WHERE ticker = ?", (current_price, ticker))
                print(f"Updated pxhw value for {ticker} to {current_price}")
                discord.post(content=f"Updated pxhw value for {ticker} to {current_price}")
            else:
                print(f"Current price is not greater than pxhw for {ticker}")
                discord.post(content=f"Current price is not greater than pxhw for {ticker}")

    else:
        # Insert a new entry for the ticker with open_trade = 0 and set last_update timestamp
        conn_o.execute("INSERT INTO oms (ticker, open_trade, last_update, txpx, pxhw) VALUES (?, ?, ?, ?, ?)",
                       (ticker, 1, current_timestamp, current_price, current_price))
        print(f"Inserted new entry for {ticker} with open_trade = 0, pxhw {current_price} and last_update timestamp")
        ems(ticker, size)

    if TEST_TRADE == 1:
        print('testing oms')
        ems(ticker, size)

    conn_o.commit()
    conn_o.close()


def act_breakout(ticker, current_price):
    connb = sqlite3.connect('breakout_history.db')
    # Get the current time and the time 60 minutes ago
    asset_name = ticker
    now = datetime.utcnow()
    sixty_minutes_ago = now - timedelta(minutes=60)

    # Query the database to check for breakouts
    queryb = "SELECT * FROM breakout_history WHERE ticker = ? AND timestamp >= ?"
    dfb = pd.read_sql(queryb, connb, params=(asset_name, sixty_minutes_ago))

    # If there was a breakout in the last 60 minutes, return True
    if len(dfb) > 0:
        print(f'Breakout already detected')
        discord_log.post(content=f'Breakout {ticker} already detected in the last 60mins...')
    else:
        print(f'Breakout {ticker}')
        discord.post(content=f'NEW Breakout {ticker}')
        try:
            cursorb = connb.cursor()
            cursorb.execute("INSERT INTO breakout_history (ticker, breakout, timestamp) VALUES (?, ?, ?)",
                            (ticker, 'DONCHIAN_HIGH', now))
            connb.commit()
            discord.post(content=f'Process OMS {ticker}...')
            oms(asset_name, current_price)
        except Exception as e:
            print(f"Error inserting breakout data for {ticker}: {e}")
            discord.post(content=f'Error act_breakout into oms {ticker}')
        print(f'Breakout {ticker}')

    connb.close()
    if TEST_TRADE == 1:
        print('testing act_breakout')
        oms(asset_name, current_price)


def calculate_donchian(df):
    period = 96  # You can adjust the period as needed
    df['donchian_channel_high'] = df['Close'].rolling(window=period).max()
    df['donchian_channel_low'] = df['Close'].rolling(window=period).min()
    df['donchian_channel_mid'] = (df['donchian_channel_high'] + df['donchian_channel_low']) / 2

def monitor_trailing(ticker, current_price):
    ticker = ticker
    current_price = current_price
    current_timestamp = datetime.now()
    conn_t = sqlite3.connect('oms.db', uri=True)
    dfo = pd.read_sql_query("SELECT * FROM oms", conn_t)

    if ticker not in dfo['ticker'].values:
        new_row = pd.DataFrame({
            'ticker': [ticker],
            'open_trade': [0],
            'last_update': [current_timestamp],
            'txpx': [current_price],
            'pxhw': [current_price]
        })
        new_row.to_sql('oms', conn_t, if_exists='append', index=False)
        print(f"Added new ticker {ticker} to the oms table.")

    # Fetch the updated DataFrame
    dfo = pd.read_sql_query("SELECT * FROM oms", conn_t)
    pxhw = dfo.loc[dfo['ticker'] == ticker, 'pxhw'].values[0]
    open_trade_value = dfo.loc[dfo['ticker'] == ticker, 'open_trade'].values[0]
    if open_trade_value == 1:
        if current_price < 0.9 * pxhw:
            print(f"Trailing Stop reached for {ticker}...")
            discord.post(content=f'Trailing Stop reached for {ticker} pxhw = {pxhw}, current price = {current_price}...')
            ctrade(ticker)
            conn_t.execute("UPDATE oms SET open_trade = 0, last_update = ?, txpx = ?, pxhw = ? WHERE ticker = ?",
                           (current_timestamp, current_price, current_price, ticker))
    conn_t.commit()
    conn_t.close()


def breakout_monitor():
    # Iterate over each ticker and fetch the current price and Donchian channels for its corresponding table
    for ticker in tickers:
        table_name = ticker
        try:
            # Fetch data from the database
            query = f"SELECT * FROM {table_name} ORDER BY ts_date DESC LIMIT 96"  # Assuming 96 rows for calculation
            df = pd.read_sql_query(query, conn)
            # Sort DataFrame by timestamp in ascending order
            df.sort_values(by='ts_date', ascending=True, inplace=True)
            calculate_donchian(df)

            # Fetch the current price from Binance
            ticker_info = client.get_symbol_ticker(symbol=f"{table_name}")
            current_price = float(ticker_info['price'])

            # Extract Donchian channels from the last row
            donchian_channel_high = df['donchian_channel_high'].iloc[-1]
            donchian_channel_mid = df['donchian_channel_mid'].iloc[-1]
            donchian_channel_low = df['donchian_channel_low'].iloc[-1]

            # Calculate the percentage difference between the database price and the current price
            db_price = df['Close'].iloc[-1]
            percentage_difference = ((current_price - db_price) / db_price) * 100

            bomy.bomy_ts_utc()
            print(
                f"Ticker: {ticker}, Current price: {current_price:.8f}, Price in DB: {db_price:.8f}, DB px close ts: {df['ts_date'].iloc[-1]}, "
                f"%-diff: {percentage_difference:.2f}%, Donchian Channel High: {donchian_channel_high:.8f}, "
                f"Donchian Channel Mid: {donchian_channel_mid:.8f},",
                f"Donchian Channel Low: {donchian_channel_low:.8f},",
                f"Donchian Rank: {round((current_price - donchian_channel_low) / (donchian_channel_high - donchian_channel_low), 2)}")

            # Check if there is a breakout and manage trailing stop
            print('monitor_trailing start')
            monitor_trailing(ticker, current_price)
            print('monitor_trailing end')
            if current_price > donchian_channel_high:
                print('### BREAKOUT ###')
                act_breakout(ticker, current_price)
            else:
                print('### NO BREAKOUT ###')

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

        print('\n')
        if TEST_TRADE == 1:
            print('testing breakout_monitor')
            print(ticker)
            act_breakout(ticker, current_price)

        if TEST_CLOSE == 1:
            print('testing trade closing')
            ctrade(ticker)
    # Close the cursor and connection when done
    cursor.close()
    conn.close()


def setup(base_url=None, skip_ws=False):
    # if '__file__' not in globals():
    # __file__ = os.getcwd() # + 'J_Workbench'
    # config_path = '/Users/michaelkilchenmann/Library/Mobile Documents/com~apple~CloudDocs/C_Code/J_Workbench/config.json'
    # config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config_path = os.path.join(os.getcwd(), "config.json")
    with open(config_path) as f:
        config = json.load(f)
    account: LocalAccount = eth_account.Account.from_key(config["secret_key"])
    address = config["account_address"]
    if address == "":
        address = account.address
    print("Running with account address:", address)
    if address != account.address:
        print("Running with agent address:", account.address)
    info = Info(base_url, skip_ws)
    user_state = info.user_state(address)
    margin_summary = user_state["marginSummary"]
    if float(margin_summary["accountValue"]) == 0:
        print("Not running the example because the provided account has no equity.")
        url = info.base_url.split(".", 1)[1]
        error_string = f"No accountValue:\nIf you think this is a mistake, make sure that {address} has a balance on {url}.\nIf address shown is your API wallet address, update the config to specify the address of your account, not the address of the API wallet."
        raise Exception(error_string)
    exchange = Exchange(account, base_url, account_address=address)
    return address, info, exchange


if __name__ == "__main__":
    breakout_monitor()
    print('...job finished...')
