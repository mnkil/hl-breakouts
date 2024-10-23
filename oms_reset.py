import sqlite3
import pandas as pd

def oms_reset():
    conn_o = sqlite3.connect('oms.db', uri=True)
    dfo = pd.read_sql_query("SELECT * FROM oms", conn_o)
    for ticker in dfo['ticker'].values:
        print(ticker)
        # dfo.loc[dfo['ticker'] == ticker, 'open_trade'].values[0] == 0
        conn_o.execute("UPDATE oms SET open_trade = 0 WHERE ticker = ?", (ticker,))

    conn_o.commit()
    conn_o.close()

if __name__ == '__main__':
    oms_reset()