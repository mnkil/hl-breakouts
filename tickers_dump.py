import yaml
from discordwebhook import Discord

def tickers_dump():

    with open("tickers.yaml", "r") as file:
        tickers = yaml.safe_load(file)["tickers"]
        print(f'tickers: {tickers}')

    with open("discord.yaml", "r") as file:
        data = yaml.safe_load(file)
        discord_url = data.get("discord_url")[0]

    discord = Discord(url=discord_url)
    discord.post(content=f"Tickers: {tickers}")

if __name__ == "__main__":
    tickers_dump()