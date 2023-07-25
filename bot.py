import argparse
import sys
from pathlib import Path
from time import sleep
import ccxt
import numpy as np
import pandas as pd
import os
import trading_functions as f
from utils import (
    get_git_revision_hash,
    get_git_revision_short_hash,
    read_credentials,
    save_config,
    str2bool,
)
from datetime import datetime, timedelta

print(os.getcwd())
import warnings
warnings.filterwarnings("ignore")
__version__ = "2022.12.21"

def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--market",
        type=str,
        default="BTC/USDT",
        help="Trading symbols, e.g. BTC/USDT - only cryptos are valid",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",#int(30 * 60 * 1000),  # (min * sec * mili)
        help="time intervall the bot trades at in milliseconds",
    )
    parser.add_argument(
        "--start_amt_in_base",
        default=0.001,
        type=float,
        help="how much of the base currency the bot should start trading with",
    )
    parser.add_argument(
        "--user",
        type=str,
        default='claus',
        help="who this bot belongs to",
    )

    args = parser.parse_args()
    config = vars(args)
    timestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config["timestr"] = timestr
    symbol = config["market"].replace("/", "-")
    config["log_pth"] = str(Path("logs") / Path(config["user"])/ Path(symbol) / timestr)
    config["base"] = symbol[: symbol.find("-")]
    config["quote"] = symbol[symbol.find("-") + 1 :]
    return config

def run_bot():
    config = get_args()
    config["version"] = __version__
    config["git_short_hash"] = get_git_revision_short_hash()
    config["git_hash"] = get_git_revision_hash()
    save_config(config)

    print("INFO: running the bot!")
    trials = 300
    config['search_params'] = {
    "search_ranges":
        { 
        "slow": {"min": 20,"max": 40},
        "fast":   {"min": 10,"max": 30},
        "smooth": {"min": 5,"max": 30},
        "signal": {"min": 5,"max": 30}
        },
    "n_trials": trials,
    "stat": "gains"
    }

    exchange_id = "binance"
    exchange_class = getattr(ccxt, exchange_id)
    exchange_settings = {"timeout": 30000, "enableRateLimit": True}

    creds = read_credentials()
    if creds is not False:
        exchange_settings = {**exchange_settings, **creds}
        creds.clear()
            
    exchange = exchange_class(exchange_settings)

    config["exchange"] = exchange

    market = config["market"]
    
    if config['timeframe'] == '1h':
        config["timeframe_in_ms"] = int(60 * 60 * 1000)
        
    timeframe_in_ms = config["timeframe_in_ms"]
    timeframe_in_sec: int = int(timeframe_in_ms / 1000)
    start_amt_in_base = config["start_amt_in_base"]
    config["base_to_sell"] = config["start_amt_in_base"]
    config["quote_to_spend"] = 0
    config['n_bots'] = 10
    symbol = config["market"].replace("/", "")

    # initialising Loggers
    event_logger = f.get_logger("event", market, config)
    config["event_logger"] = event_logger
    pid = os.getpid()
    pid_logger = f.get_logger(f'PID {pid}', market, config)
    pid_logger.info(f'PID {pid}')
    config['asset_start_price'] = f.get_price("avg_price", config)
    balance_start = exchange.fetch_balance()

    print(sum(balance_start['total'].values()))

    event_logger.info(str(config))
    event_logger.info(f'total account value in USDT: {sum(balance_start["total"].values())}')

    ########################################################################################################################
    ## SANITY CHECK TRADE AMOUNTS
    ########################################################################################################################

    available_base = f.fetch_base(config)

    # enough total base available?
    if available_base < start_amt_in_base:
        event_logger.info(
            f"Not enough base available to start trading. You requested {start_amt_in_base} but only have {available_base}. Quitting now."
        )
        sys.exit()

    # start_amt_in_base big enough?
    min_base_order_amount = f.get_min_base_order_amount(config)

    if min_base_order_amount > start_amt_in_base:
        event_logger.info(
            f"Base amount  too small. You requested {start_amt_in_base} but need at least {min_base_order_amount}. Quitting now."
        )
        sys.exit()

    event_logger.info(
        f"{np.round(((min_base_order_amount/start_amt_in_base)-1)*100,2)} % drawdown away from not being able to trade."
    )

    ########################################################################################################################
    ## INITIALIZE DATAFRAME
    ########################################################################################################################

    now = exchange.milliseconds()
    event_logger.info(f"Initializing")
    # training history 
    earliest_timestamp_in_ms = f.get_earliest(exchange,market)

    # training history should end 1h before now, so the first candle in the while loop completes the history
    config['hist_start_ms'] = earliest_timestamp_in_ms#+(1000*60*60*24*365*5)# for testing
    config['hist_end_ms'] = int(now - timeframe_in_ms) # in milliseconds
    config['last_order'] = 'buy'

    df = f.init_df(config)

    config['param_list'] = f.get_new_param_list(df,config)
    event_logger.info(config['param_list'])

    event_logger.info("Bot initialized")
    event_logger.info(f"start price {f.get_price('avg_price', config)}")
    event_logger.info(f"start time: {datetime.now().strftime('%d.%m.%Y - %H:%M:%S')}")

    ########################################################################################################################
    ## WHILE LOOP
    ########################################################################################################################
    first_loop = True
    next_row = None
    counter = 1
    print("entering while-loop")
    while True:
        
        loop_time = datetime.now()

        if counter%500==0: # retrain every 20 days
            config['param_list'] = f.get_new_param_list(df,config)
            event_logger.info(config['param_list'])
            next_row = len(df)
            df.loc[next_row,"retraining"] = 1
        else:
            next_row = len(df)
            
        if first_loop:
            next_row = len(df)-1
            first_loop = False

        config["next_row"] = next_row 

        #############################
        ## UPDATE DF AND RUN STRATEGY
        #############################

        df = f.update_price_info(df, config)
        df = f.bagging(df, config)

        ########################################################################################################################
        ##    O    R    D    E    R    S
        ########################################################################################################################

        if df.loc[next_row,"position"] == 'short':
            if config['last_order'] == 'buy':
                df = f.trigger_sell_order(df, config)
                f.log_order("sell", df, config)
                config['last_order'] = 'sell'

        # fast over slow --> buy
        elif df.loc[next_row,"position"] == 'long':
            if config['last_order'] == 'sell':               
                df = f.trigger_buy_order(df, config)
                f.log_order("buy", df, config)
                config['last_order'] = 'buy'

        df = f.update_amounts(df, config)

        df_name = f"df_{symbol}.csv"
        df_path = str(Path(config["log_pth"]) / df_name)
        df.to_csv(df_path, sep=";")
        
        # wait until one timeframe has passed
        # use while loop top account for delays
        while (datetime.now() - loop_time) <  timedelta(seconds=timeframe_in_sec):
            sleep(5)
        counter +=1


if __name__ == "__main__":
    run_bot()
