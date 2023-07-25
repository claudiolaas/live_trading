
#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import streamlit as st
import pandas as pd
import numpy as np
from sys import platform
import plotly.io as pio
pio.renderers.default='browser'

st.set_page_config(layout="wide")
if 'win' in platform:
    logs_dir = Path(r'C:\Users\claud\OneDrive\Documents\python\tb\live_trading\logs')
else:
    logs_dir = Path(r'logs')

if 'win' in platform:
    user_dir = Path(r'C:\Users\claud\OneDrive\Documents\python\tb\live_trading\logs\claus')
else:
    user_dir = Path(r'logs/claus')

users = [f.name for f in logs_dir.iterdir() if f.is_dir()]
user = st.radio('users',users)
user_dir = Path(logs_dir,user)

bots = [f.name for f in user_dir.iterdir() if f.is_dir()]
bot_dir = 'BTC-USDT'
bot_dir = st.radio('bots',bots)

base_dir = Path(user_dir,bot_dir)

all_subdirs = [Path(base_dir,d) for d in os.listdir(base_dir)]
newest_bot = max(all_subdirs, key=os.path.getmtime)
st.write(newest_bot)
df = pd.read_csv(
        Path(newest_bot,'df_BTCUSDT.csv'),
        delimiter=";",
    )
start_index = df["buy_price"].first_valid_index()


df = df.loc[start_index:,:].reset_index().copy(deep=True)
df.loc[0,"bot_return"] = df.loc[0,"close"]

close = go.Candlestick(x=df['dt'].astype(np.datetime64),
                open=df["close"].shift(),
                high=df["close"],
                low=df["close"],
                close=df["close"])

hist = go.Scatter(x=df["dt"], y=df["hist"], name="Signal", mode="lines")
balance = go.Scatter(x=df["dt"], y=df["wallet_balance_in_usdt"], name="Balance", mode="lines")

buy = go.Scatter(
    x=df["dt"],
    y=df["buy_price"],
    name="Buys",
    mode="markers",
    marker=dict(color="Green", size=15),
)

sell = go.Scatter(
    x=df["dt"],
    y=df["sell_price"],
    name="Sells",
    mode="markers",
    marker=dict(color="Red", size=15),
)

bot_eq = go.Scatter(
    x=df["dt"],
    y=df["bot_return"].cumprod(),
    name="Bot Return",
    mode="lines",
    line = {'width' : 4}
        
)

value_in_quote = go.Scatter(
    x=df["dt"],
    y=df["total_bot_value_in_quote"],
    name="Bot Value in quote",
    mode="lines",
    line = {'width' : 4}
        
)

total_return_bot = df["bot_return"].cumprod().iloc[-1]/df["bot_return"].cumprod().iloc[0]
total_return_asset = df["asset_return"].cumprod().iloc[-1]/df["asset_return"].cumprod().iloc[0]

st.write(f'Total return:')
st.write(f'Bot: {np.round(total_return_bot*100,2)} %')
st.write(f'Asset: {np.round(total_return_asset*100,2)} %')


fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "Performance",
        "Signal",
        'Balance'
    ),
)
# 1
fig.add_trace(close, row=1, col=1)
fig.add_trace(buy, row=1, col=1)
fig.add_trace(sell, row=1, col=1)

fig.add_trace(bot_eq, row=1, col=1)

# 2
fig.add_trace(hist, row=2, col=1)

fig.update_traces(xaxis="x1")
fig.update_layout(hovermode="x unified")
# 3
fig.add_trace(value_in_quote, row=3, col=1)
# fig.add_trace(balance, row=3, col=1)
# 
fig.update_layout(height=2500)
fig.update_layout(autosize=True)

fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(template='simple_white')

fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid')

fig.update_xaxes(showspikes=True,  spikesnap='cursor', spikedash='solid')

st.plotly_chart(fig,use_container_width=True)

# %%
