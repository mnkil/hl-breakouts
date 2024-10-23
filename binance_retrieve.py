from datetime import datetime
import pandas as pd
import sqlite3
import bomy
import yaml
from discordwebhook import Discord
import os

with open("discord.yaml", "r") as file:
    data = yaml.safe_load(file)
    discord_url = data.get("discord_url")[0]
    client1 = data.get("client1")[0]
    client2 = data.get("client2")[0]
    client2 = data.get("client2")[0]
    discord_log_url = data.get("discord_log_url")[0]

discord = Discord(url=discord_url)
discord_log = Discord(url=discord_log_url)
discord_log.post(content=f'Data Retrieve - DB: {os.getcwd()}...')

def fetch_missing_data(tickers, conn):
    now = datetime.utcnow()
    end_ts = pd.to_datetime(now)

    for i, ticker in enumerate(tickers, 1):
        # Check if the table exists in the database
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (ticker,))
        table_exists = cursor.fetchone() is not None
        cursor.close()

        if not table_exists:
            # If the table doesn't exist, set latest_timestamp to None
            latest_timestamp = None
        else:
            # Check the latest timestamp in the database for the current ticker
            query = f"SELECT MAX(ts_date) FROM {ticker}"
            latest_timestamp = pd.read_sql_query(query, conn).iloc[0, 0]

        if pd.isnull(latest_timestamp):
            start_date = '2023-09-01'  # Assuming a starting date for initial data retrieval
        else:
            start_date = latest_timestamp  # Start from the latest timestamp in the database

        # Retrieve data from the latest timestamp in the database up to now
        df = bomy.ret_binance_spot(ticker, '1H', start_date, end_ts)

        # Append the new data to the database
        df.to_sql(name=ticker, con=conn, if_exists='append', index=False)

        # Log progress and completion percentage
        print(f"Data updated for {ticker} ({i}/{len(tickers)})")
        completion_percentage = (i / len(tickers)) * 100
        print(f"Progress: {completion_percentage:.2f}%")

        # Log message using bomy_ts
        bomy.bomy_ts(f"Data updated for {ticker}")


def main():
    # Read tickers from YAML file

    try:
        with open("/home/ubuntu/tickers.yaml", "r") as file:
            tickers = yaml.safe_load(file)["tickers"]
    except:
        with open("tickers.yaml", "r") as file:
            tickers = yaml.safe_load(file)["tickers"]

    # Connect to SQLite database
    conn = sqlite3.connect('historical_data.db')

    # Fetch missing data for each ticker
    fetch_missing_data(tickers, conn)

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()
