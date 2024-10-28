# pyright: strict

import datetime
from typing import List, NamedTuple, Optional

import pandas as pd
from pandas import Timestamp

from mlib.core.time_utils import get_ts


class ExchangeConfig(NamedTuple):
    symbols: List[str]
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open_auction_start_time: Optional[pd.Timestamp]  # open call auction
    open_auction_end_time: Optional[pd.Timestamp]
    continuous_auction_start_time: Optional[pd.Timestamp]
    continuous_auction_end_time: Optional[pd.Timestamp]
    close_auction_start_time: Optional[pd.Timestamp]  # close call auction
    close_auction_end_time: Optional[pd.Timestamp]


def create_Chinese_stock_exchange_config(date: datetime.date, symbols: List[str]):
    config = ExchangeConfig(
        symbols=symbols,
        open_time=get_ts(date, 9, 10, 0),
        close_time=get_ts(date, 15, 0, 0),
        open_auction_start_time=get_ts(date, 9, 15, 0),
        open_auction_end_time=get_ts(date, 9, 25, 0),  # match at 9:25, all open call auction orders are in [9:15, 9:25)
        close_auction_start_time=get_ts(date, 14, 57, 0),
        close_auction_end_time=get_ts(
            date, 15, 0, 0
        ),  # match at 15:00, all close call auction orders are in [14:57, 15, 00)
        continuous_auction_start_time=get_ts(date, 9, 30, 0),
        continuous_auction_end_time=get_ts(date, 14, 56, 59, 999999),
    )
    return config


def create_exchange_config_without_call_auction(market_open: Timestamp, market_close: Timestamp, symbols: List[str]):
    config = ExchangeConfig(
        symbols=symbols,
        open_time=market_open,
        close_time=market_close,
        open_auction_start_time=None,
        open_auction_end_time=None,
        close_auction_start_time=None,
        close_auction_end_time=None,
        continuous_auction_start_time=market_open,
        continuous_auction_end_time=market_close,
    )
    return config
