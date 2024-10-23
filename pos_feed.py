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
import json

# info = Info(constants.TESTNET_API_URL, skip_ws=True)
info = Info(constants.MAINNET_API_URL, skip_ws=True)
user_state = info.user_state("0xcECE8E0C2eBED9bc18CE2184fF87b91E7FE33022")
# print(user_state)

# Read tickers from YAML file
with open("tickers.yaml", "r") as file:
    tickers = yaml.safe_load(file)["tickers"]
    print(f'tickers: {tickers}')

with open("oms.yaml", "r") as file:
    notional_usd = yaml.safe_load(file)["notional_usd"][0]

with open("discord.yaml", "r") as file:
    data = yaml.safe_load(file)
    discord_url = data.get("discord_url")[0]
    client1 = data.get("client1")[0]
    client2 = data.get("client2")[0]


print('### POSITION FEED ###')
print('\n')

# discord webhook
discord = Discord(url=discord_url)
client = Client(client1, client2)

def feedr():
    # Load user_state JSON into a Python dictionary
    user_state_dict = json.loads(json.dumps(user_state))

    # Accessing specific values and rounding them
    account_value = round(float(user_state_dict['marginSummary']['accountValue']))
    total_ntl_pos = round(float(user_state_dict['marginSummary']['totalNtlPos']))
    cross_margin_used = round(float(user_state_dict['crossMaintenanceMarginUsed']))
    withdrawable = round(float(user_state_dict['withdrawable']))

    # Create a formatted message
    message = (
        f'Account Value: {account_value:,}\n'
        f'Total Notional Position: {total_ntl_pos:,}\n'
        f'Cross Maintenance Margin Used: {cross_margin_used:,}\n'
        f'Withdrawable: {withdrawable:,}\n\n'
    )

    # Accessing asset positions and rounding values
    for position in user_state_dict['assetPositions']:
        coin = position['position']['coin']
        position_value = round(float(position['position']['positionValue']))
        unrealized_pnl = round(float(position['position']['unrealizedPnl']))
        message += f'{coin}: Position Value: {position_value:,}, Unrealized PnL: {unrealized_pnl:,}\n'

    # Print the message to the console
    print(message)

    # Send the message to Discord
    discord.post(content=message)

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
    feedr()
    print('\n### FEED COMPLETE ###')