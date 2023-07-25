#%%
import math
from shutil import ExecError
import sys
from datetime import datetime as dt
from pathlib import Path
from time import sleep
from typing import Any, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tenacity import retry, stop_after_attempt, wait_fixed
import optuna
from optuna.samplers import GridSampler, NSGAIISampler, RandomSampler, TPESampler,CmaEsSampler

from utils import get_logger, send_email
import itertools


#%%


def truncate(n: float, decimals: int) -> float:
    # TODO: truncate as string
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


@retry(wait=wait_fixed(20),stop=stop_after_attempt(5))
def fetch_balance(exchange, logger, symbol: str, selector: str = None) -> float:
    balance: float = 0.0

    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        message = f"could not fetch initial base balance due to {e}"
        logger.error(message)

    try:
        if selector is not None:
            balance = balance[symbol][selector]
        else:
            balance = balance[symbol]
    except Exception as e:
        err_msg = f"{symbol} not found in balance"
        logger.error(err_msg)

    return balance


@retry(wait=wait_fixed(20),stop=stop_after_attempt(5))
def fetch_base(config) -> float:
    base: float = 0.0

    try:
        base = config["exchange"].fetch_balance()
    except Exception as e:
        message = f"could not fetch base balance due to {e}"
        config["event_logger"].error(message)

    try:
        base = base[config["base"]]["free"]

    except Exception as e:
        message = f"could not fetch free base due to {e}"
        config["event_logger"].error(message)

    return base


@retry(wait=wait_fixed(20),stop=stop_after_attempt(5))
def fetch_quote(config) -> float:
    quote: float = 0.0
    try:
        quote = config["exchange"].fetch_balance()
    except Exception as e:
        message = f"could not fetch quote balance due to {e}"
        config["event_logger"].error(message)

    try:
        quote = quote[config["quote"]]["free"]

    except Exception as e:
        message = f"could not fetch free quote due to {e}"
        config["event_logger"].error(message)

    return quote

def update_amounts(df: pd.DataFrame, config: dict) -> dict:

    next_row = config['next_row']
    df.loc[next_row, 'quote_to_spend'] = config['quote_to_spend']
    df.loc[next_row, 'base_to_sell'] = config['base_to_sell'] 

    df.loc[next_row,"quote_after_order"] = fetch_quote(config)
    df.loc[next_row,"base_after_order"] = fetch_base(config)

    df.loc[next_row,"delta_quote"] =  df.loc[next_row,"quote_after_order"] - df.loc[next_row,"quote_before_order"]
    df.loc[next_row,"delta_base"] =  df.loc[next_row,"base_after_order"] - df.loc[next_row,"base_before_order"] 

    value_before = df.loc[next_row,"base_before_order"] * df.loc[next_row,"close"] + df.loc[next_row,"quote_before_order"]
    value_after = df.loc[next_row,"base_after_order"] * df.loc[next_row,"close"] + df.loc[next_row,"quote_after_order"]

    df.loc[next_row,"value_before"]  = value_before
    df.loc[next_row,"value_after"]  = value_after

    df.loc[next_row,"value_delta_in_quote"] = value_before - value_after

    df.loc[next_row,"total_base"] = config["exchange"].fetch_balance()[config["base"]]['total']
    df.loc[next_row,"total_quote"] = config["exchange"].fetch_balance()[config["quote"]]['total']


    # log total asset return
    total_asset_return = df.loc[next_row,"close"] / config['asset_start_price']
    total_asset_return_percent = (total_asset_return - 1) * 100
    df.loc[next_row,"total_asset_return_percent"] = np.round(total_asset_return_percent,4)
    
    # log total bot value
    total_bot_value_in_quote = config['base_to_sell']  * df.loc[next_row,"close"] + config['quote_to_spend']
    df.loc[next_row,"total_bot_value_in_quote"] = total_bot_value_in_quote
    df["bot_return"] = df["total_bot_value_in_quote"]/df["total_bot_value_in_quote"].shift()

    df.loc[next_row,"wallet_balance_in_usdt"] = get_wallet_balance_in_usdt(config)
    
    # log total bot return
    total_start_value_in_quote = config["start_amt_in_base"] * config['asset_start_price']
    
    total_bot_return = total_bot_value_in_quote / total_start_value_in_quote
    total_bot_return_percent = (total_bot_return - 1) * 100
    df.loc[next_row,"total_bot_return_percent"] = np.round(total_bot_return_percent,4)

    # log alpha
    alpha_percent = total_bot_return_percent - total_asset_return_percent
    df.loc[next_row,"alpha_percent"] = np.round(alpha_percent,4)


    return df


def trigger_sell_order(df: pd.DataFrame, config: dict) -> dict:
    next_row = config['next_row']

    config["event_logger"].info(f'trying to sell amount {config["base_to_sell"]}')
    config["precice_base_to_sell"] = float(config["exchange"].amount_to_precision(config["market"], config["base_to_sell"]))

    sell_order = send_sell_order(df, config)

    print(sell_order)

    df.loc[next_row,"sell_price"] = sell_order['average']
    df.loc[next_row,"average"] = sell_order['average']
    df.loc[next_row,"cost"] = sell_order['cost']
    df.loc[next_row,"amount"] = sell_order['amount']

    config["quote_to_spend"] = config["quote_to_spend"] + sell_order['cost']
    config["base_to_sell"] = config["base_to_sell"] - sell_order['amount']

    config['last_sell_price'] =  sell_order['average']
    config['sell_order'] = sell_order

    return df

@retry(wait=wait_fixed(20),stop=stop_after_attempt(5))
def send_sell_order(df: pd.DataFrame, config: dict) -> dict:

    try:
        sell_order = config["exchange"].create_order(
            symbol=config["market"],
            type="market",
            side="sell",
            amount=config["precice_base_to_sell"]
            )
        
        sleep(1)
        # re-assign otherwise the order will be diplayed as 'not filled'
        sell_order = config["exchange"].fetchOrder(sell_order["id"],symbol = config['market'])
        while not sell_order['status']=='closed':
            sell_order = config["exchange"].fetchOrder(sell_order["id"],symbol=config['market'])
            config["event_logger"].error('waiting for order fill')
            sleep(1)
        return sell_order

    except Exception as e:
        message = f"Bot in {config['market']} failed to sell {config['precice_base_to_sell']} base due to {e}"
        config["event_logger"].error(message)
        config["event_logger"].error('retrying')

        raise ValueError(message)

def trigger_buy_order(df: pd.DataFrame, config: dict) -> dict:
    
    next_row = config['next_row']

    config["event_logger"].info(f'trying to buy amount {config["quote_to_spend"] / get_price("ask", config)}')

    buy_order = send_buy_order(df, config)
    
    print(buy_order)
    
    df.loc[next_row,"average"] = buy_order['average']
    df.loc[next_row,"buy_price"] = buy_order['average']
    df.loc[next_row,"cost"] = buy_order['cost']
    df.loc[next_row,"amount"] = buy_order['amount']

    config["quote_to_spend"] = config["quote_to_spend"] - buy_order['cost']
    config["base_to_sell"] = config["base_to_sell"] + buy_order['amount']

    config['last_buy_price'] = buy_order['average']
    config['buy_order'] = buy_order

    return df

@retry(wait=wait_fixed(20),stop=stop_after_attempt(5))
def send_buy_order(df: pd.DataFrame, config: dict) -> dict:

    try:
        # buy_amount = df.loc[next_row,"quote_to_spend"] / get_price("ask", config)
        buy_order = config["exchange"].create_order(
            symbol=config["market"],
            type="market",
            amount=config["quote_to_spend"],
            side="buy",
            price = None,
            params = {
                'quoteOrderQty': config["quote_to_spend"],
            }
        )
        # re-assign otherwise the order will be diplayed as 'not filled'
        buy_order = config["exchange"].fetchOrder(buy_order["id"],symbol=config['market'])
        while not buy_order['status']=='closed':
            buy_order = config["exchange"].fetchOrder(buy_order["id"],symbol=config['market'])
            config["event_logger"].error('waiting for order fill')
            sleep(1)
        return buy_order

    except Exception as e:
        message = f"Bot in {config['market']} failed to buy {config['base']} for {config['quote_to_spend']} USDT due to {e}"
        config["event_logger"].error(message)
        config["event_logger"].error('retrying')

        raise ValueError(message)

def update_price_info(df: DataFrame, config: dict) -> pd.DataFrame:
    next_row = config['next_row']
    current_price = get_price("avg_price", config)
    ask = get_price("ask", config)
    bid = get_price("bid", config)
    current_spread = get_price("spread", config)
    spread_in_percent = (current_spread - 1) * 100

    df.loc[next_row,"close"] = current_price
    df.loc[next_row,"ask"] = ask
    df.loc[next_row,"bid"] = bid
    df.loc[next_row,"spread_in_percent"] = spread_in_percent
    df.loc[next_row,"quote_before_order"] = fetch_quote(config)
    df.loc[next_row,"base_before_order"] = fetch_base(config)

    # get current time
    df.loc[next_row,"dt"] = config["exchange"].iso8601(config["exchange"].milliseconds())

    df["asset_return"] = df["close"] / df["close"].shift()
    df["market"] = config["market"]

    return df

trials = 100
search_params = {
    "search_ranges":
        { 
        "slow": {"min": 10,"max": 40},
        "fast":   {"min": 5,"max": 30},
        "smooth": {"min": 1,"max": 15},
        "signal": {"min": 2,"max": 15}
        },
    "n_trials": trials,
    "stat": "gains"
}

def objective(trial, strategy, data, search_ranges, stat):

    strategy_params = {}
    for k, v in search_ranges.items():
        grid_point = trial.suggest_int(k, v["min"], v["max"])
        strategy_params[k] = grid_point
    
    if strategy_params['fast'] >= strategy_params['slow']:
        return 0
    else:
        df = strategy(data, strategy_params)
        df = get_returns(df)
        stats = get_stats(df['bot_return'],prnt=False)

        return stats[stat]  

def logging_callback(study: optuna.Study, frozen_trial):
    if study.user_attrs.get("previous_best_value", 0) < study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "{} - {}x - {}".format(
                frozen_trial.number,
                np.round(frozen_trial.value, 1),
                frozen_trial.params,
            )
        )
    if frozen_trial.number % 500 == 0:
        print(
            "{} - {}x - {}".format(
                frozen_trial.number,
                np.round(frozen_trial.value, 1),
                frozen_trial.params,
            )
        )

def search(df, strategy, search_params):
    # study = optuna.create_study(sampler=TPESampler(), direction="maximize")
    study = optuna.create_study(sampler=CmaEsSampler(restart_strategy='ipop', inc_popsize=2), direction="maximize")

    study.optimize(
        lambda trial: objective(
            trial,
            strategy=strategy,
            data=df,
            search_ranges=search_params["search_ranges"],
            stat=search_params["stat"],
        ),
        # callbacks=[logging_callback],
        n_trials=search_params["n_trials"],
    )

    study_df = study.trials_dataframe()

    df = strategy(df, study.best_params)
    df = get_returns(df)
    stats = get_stats(df['bot_return'],prnt=False)
    stats['best_params'] = study.best_params

    return stats,study_df

def get_stats(returns: pd.Series, prnt: bool = False) -> dict:
    df = pd.DataFrame({"returns": returns}).dropna()
    df = df.dropna().reset_index(drop=True)
    stats = {}

    df["eq"] = df["returns"].cumprod()
    gains = np.round(df["eq"].tail(1).values[0] / df["eq"].head(1).values[0], 3)
    length = len(df)
    dca = df["eq"].mean()
    years = len(df) / 8760  # 8760 Stunden im Jahr
    cagr = np.round(np.power(gains, (1 / years)), 3)

    sharpe = df['returns'].mean() / df['returns'].std()

    sharpe_yearly = np.round(np.power(sharpe,(1/years)),5)
    max_dd_perc, max_dd_ix = calc_max_drawdown(df["eq"])

    stats["cagr"] = cagr
    stats["drawdown"] = max_dd_perc
    stats["gains"] = gains
    stats["lenght"] = length
    stats["dca"] = dca
    stats['sharpe'] = sharpe_yearly

    if prnt:
        print(stats)

    return stats

def calc_max_drawdown(eq_curve: list) -> Tuple[float, float]:
    df = pd.DataFrame({"eq_curve": eq_curve})
    df["max"] = df["eq_curve"].expanding().max()
    df["drawdown"] = df["eq_curve"] / df["max"]

    max_dd_dec = df["drawdown"].min() - 1
    max_dd_perc = f"{np.round(max_dd_dec*100,2)} %"
    max_dd_ix = int(
        df["drawdown"].astype(float).argmin()
    )  # int() to make it JSON serializable

    return max_dd_perc, max_dd_ix

def init_df(config):

    df = get_df(config['hist_start_ms'], config['hist_end_ms'], config)
    next_row = len(df)

    start_price = get_price("avg_price", config)

    next_row = len(df)
    df.loc[next_row,'base_to_sell'] = config["start_amt_in_base"]
    df.loc[next_row,'quote_to_spend'] = 0
    df.loc[next_row,'start_price'] = start_price
    df.loc[next_row,'sell_price'] = start_price
    df.loc[next_row,'buy_price'] = start_price
    config['last_sell_price'] = start_price
    config['last_buy_price'] = start_price
    df.loc[next_row,'order_object'] = None
    df.loc[next_row,'delta_quote'] = 0
    df.loc[next_row,'delta_base'] = 0


    # save df
    symbol = config["market"].replace("/", "")
    df_name = f"df_{symbol}.csv"
    df.to_csv(Path(config["log_pth"]) / df_name)
    return df

def get_earliest(exchange,market):

    since = exchange.parse8601('2010-01-01' + "T00:00:00Z")
    until = exchange.parse8601('2050-01-01' + "T00:00:00Z")
    while since < until:
        orders = exchange.fetchOHLCV(market, timeframe='1M', since=since)
        if len(orders)>0:
            earliest = orders[0][0]
            return earliest
        else:
            since+=(1000*60*60*24*30)

def get_df(start: int, end: int, config: dict) -> pd.DataFrame:

    all_candles = []
    while start < end:
        candles = config["exchange"].fetchOHLCV(
            config["market"], timeframe="1h", since=start
        )
        if len(candles):
            start = candles[-1][0]
            all_candles += candles
        else:
            break

    df = pd.DataFrame(all_candles)
    df.drop_duplicates(inplace=True)
    df.rename(
        columns={0: "dt", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"},
        inplace=True,
    )
    df["dt"] = df["dt"].apply(lambda x: config["exchange"].iso8601(x))
    df = df.iloc[:, [0, 4]].reset_index(drop=True)

    # needed for plotting
    df['buy_price'] = np.nan
    df['sell_price'] = np.nan 

    return df


@retry(wait=wait_fixed(20),stop=stop_after_attempt(5))
def get_price(selector, config) -> float:
    """Fetched den aktuellen Marktpreis für die jeweilige Seite"""

    try:
        orderbook = config["exchange"].fetch_order_book(config["market"])
    except Exception as e:
        if config["event_logger"]:
            config["event_logger"].error(
                f"get_price failed, retrying. Exception: {e}"
            )
        else:
            print(f"get_price failed, retrying: {e}")

    # TODO: check if tenacity retries because of the handled exception or because
    # of the unhandled KeyError below (orderbook is unbound)
    bid: float = orderbook["bids"][0][0] if len(orderbook["bids"]) > 0 else None
    ask: float = orderbook["asks"][0][0] if len(orderbook["asks"]) > 0 else None
    spread: float = (ask / bid) if (bid and ask) else None
    avg_price: float = (bid + ask) / 2
    bid_ask: dict[str, float] = {
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "avg_price": avg_price,
    }
    return bid_ask[selector]


def get_returns(
    df,
    price_col: str = "close",
    position_col: str = "position",
    suffix: str = "",
    fee: float = 0.00005,
    slippage: float = 0.0000,
) -> pd.DataFrame:

    # nan might appear if slow==fast, in that case repeat last position
    df[position_col] = df[position_col].fillna(method="ffill")

    buys = (df.loc[:, position_col].shift(1) == "short") & (
        df.loc[:, position_col] == "long"
    )
    sells = (df.loc[:, position_col].shift(1) == "long") & (
        df.loc[:, position_col] == "short"
    )
    # put fee at every order, otherwise 1
    df[f"trade{suffix}"] = np.where((buys | sells), 1, 0)
    df[f"fee{suffix}"] = np.where((buys | sells), 1 - fee, 1)

    # calculate slippage
    conditions = [buys, sells]
    choices = [df[price_col] * (1 + slippage), df[price_col] * (1 - slippage)]

    df[f"slipped_close{suffix}"] = np.select(conditions, choices, default=df[price_col])

    df[f"asset_return{suffix}"] = df[f"slipped_close{suffix}"] / df[f"slipped_close{suffix}"].shift()

    conditions = [
        (df[position_col].shift() == "long"),
        (df[position_col].shift() == "short"),
    ]
    choices = [df["asset_return"], 1]

    df[f"bot_return{suffix}"] = (
        np.select(conditions, choices, default=np.nan) * df[f"fee{suffix}"]
    )
    df[f"buy_prices{suffix}"] = np.where(buys, df[f"slipped_close{suffix}"], np.nan)
    df[f"sell_prices{suffix}"] = np.where(sells, df[f"slipped_close{suffix}"], np.nan)

    df[f"bot_eq_curve{suffix}"] = df[f"bot_return{suffix}"].cumprod()

    return df

    # position | position.shift() | asset_return | bot_return * fee
    # long               na              0.95           1         1
    # long               long            0.99           0.99      1
    # long               long            1.01           1.01      1
    # short              long            0.88           0.88      0.99925
    # short              short           0.95           1         1
    # long               short           1.05           1         0.99925
    # long               long            1.02           1.02      1

def get_new_param_list(df,config):
    
    df_train = df.reset_index(drop=True).copy(deep=True)
    train_stats, study_df = search(df_train, macd, config['search_params'])

    # get n best bots MACD
    cols = ['value','params_fast', 'params_signal', 'params_slow', 'params_smooth']
    top_params = study_df[cols].drop_duplicates().sort_values('value',ascending=False).reset_index().head(config['n_bots'])

    param_list = []
    for ix, row in top_params.iterrows():
        param_list.append(
            {'slow': int(row['params_slow']), 
                'fast': int(row['params_fast']), 
                'smooth': int(row['params_smooth']), 
                'value': row['value'], 
                'signal': int(row['params_signal'])})

    return param_list

def bagging(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # run all bots over df_test and combine indicator by performance weighted average
    hists = []
    rs = []
    for param in config['param_list']:
        df_temp = macd(df.copy(),param)
        df_temp = get_returns(df_temp)
        rs.append(df_temp['bot_return'].prod())
        hists.append(df_temp['hist']*param['value'])

    combined_hists = pd.concat(hists,axis=1)
    combined_hists = combined_hists.sum(axis=1)/sum([x['value'] for x in config['param_list']])
    df['temp_hist'] = combined_hists

    # only write last value of temp_hist over to hist so that the history of values does not get overwritte when retraining
    df.loc[config['next_row'],'hist'] = df.loc[config['next_row'],'temp_hist']
    df['temp_position'] = np.where(df['hist'].diff() < 0, 'short', 'long')
    df.loc[config['next_row'],'position'] = df.loc[config['next_row'],'temp_position']

    return df

def macd(df,params):

    df["smooth"] = df["close"].rolling(params['smooth']).mean()
    df["fast"] = df["smooth"].rolling(params['fast']).mean()
    df["slow"] = df["smooth"].rolling(params['slow']).mean()
    df['macd'] = df['fast'] - df['slow']
    df["signal"] = df["macd"].rolling(params['signal']).mean()
    df['hist'] = df['macd']-df['signal']

    df['long'] = 'long'
    df['short'] = 'short'

    df['position'] = np.where(df['hist'].diff()<0,'short','long',)

    return df


def update_indicator(df: pd.DataFrame, config: dict) -> pd.DataFrame:

    df["smoothed"] = df["close"].rolling(window=config["smoothing"]).mean()
    df["fast"] = df["smoothed"].rolling(window=config["fast_lookback"]).mean()
    df["slow"] = df["smoothed"].rolling(window=config["slow_lookback"]).mean()
    df['macd'] = df['fast'] - df['slow']
    df["signal"] = df["macd"].rolling(window=config["signal"]).mean()

    df["fast_ma"] = df["smoothed"].rolling(window=config["fast_lookback"]).mean()
    df["slow_ma"] = df["smoothed"].rolling(window=config["slow_lookback"]).mean()

    return df


def get_min_base_order_amount(config) -> float:

    price = get_price("avg_price", config)

    if config["quote"] == "USD":
        return 0.001

    if config["quote"] == "USDT":
        return 0.0001

    elif config["quote"] == "BNB":
        return 0.05 / price

    elif config["quote"] == "BTC":
        return 0.0001 / price

    elif config["quote"] == "ETH":
        return 0.005 / price

    elif config["quote"] == "BUSD":
        return 10 / price
    else:
        config["event_logger"].info(f"Unknown quote currency {config['quote']}. Quitting now.")
        sys.exit()


def log_order(type, df, config):

    next_row = config['next_row']
    order_return = np.round(((config['last_sell_price'] / config['last_buy_price']) - 1) * 100, 4)

    amount = df.loc[next_row,"amount"]
    cost = df.loc[next_row,"cost"]
    price = np.round(df.loc[next_row,"average"], 3)

    if type == "sell":
        config["event_logger"].info(
            f"SELL - amount: {amount} {config['base']} - cost: {cost} {config['quote']} - price: {price} {config['market']} - return: {order_return} %"
        )
    elif type == "buy":
        config["event_logger"].info(
            f" BUY - amount: {amount} {config['base']} - cost: {cost} {config['quote']} - price: {price} {config['market']} - return: {order_return} %"
        )

def get_wallet_balance_in_usdt(config):

    balances = config['exchange'].fetch_balance()['info']['balances']
    balance = 0

    for i in balances:
        if float(i['free'])!=0 or float(i['locked'])!=0:
            if i['asset'] != 'USDT':
                request = i['asset']+'/USDT'
                try:
                    price = float(config['exchange'].fetchTicker(request)['last'])
                except:
                    try:
                        request = i['asset']+'/BTC'
                        price = float(config['exchange'].fetchTicker(request)['last'])
                        val_btc = (float(i['free'])+float(i['locked']))*price
                        price = float(config['exchange'].fetchTicker('BTC/USDT')['last'])
                        balance += val_btc*price
                    except:
                        try:
                            request = i['asset']+'/ETH'
                            price = float(config['exchange'].fetchTicker(request)['last'])
                            val_btc = (float(i['free'])+float(i['locked']))*price
                            price = float(config['exchange'].fetchTicker('ETH/USDT')['last'])
                            balance += val_btc*price
                        except:
                            pass

                balance += (float(i['free'])+float(i['locked']))*price
            else:
                balance += float(i['free'])+float(i['locked'])
    
    return balance
