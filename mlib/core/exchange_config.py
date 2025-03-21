# pyright: strict

import datetime
from typing import NamedTuple

import pandas as pd
from pandas import Timestamp

from mlib.core.time_utils import get_ts


class ExchangeConfig(NamedTuple):
    """Exchange configuration for a specific market."""

    symbols: list[str]
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open_auction_start_time: pd.Timestamp | None  # open call auction
    open_auction_end_time: pd.Timestamp | None
    continuous_auction_start_time: pd.Timestamp | None
    continuous_auction_end_time: pd.Timestamp | None
    close_auction_start_time: pd.Timestamp | None  # close call auction
    close_auction_end_time: pd.Timestamp | None


def create_Chinese_stock_exchange_config(date: datetime.date, symbols: list[str]) -> ExchangeConfig:  # noqa: N802
    """Create exchange configuration for Chinese stock exchanges."""
    config = ExchangeConfig(
        symbols=symbols,
        open_time=get_ts(date, 9, 10, 0),
        close_time=get_ts(date, 15, 0, 0),
        open_auction_start_time=get_ts(date, 9, 15, 0),
        open_auction_end_time=get_ts(date, 9, 25, 0),  # match at 9:25, all open call auction orders are in [9:15, 9:25)
        close_auction_start_time=get_ts(date, 14, 57, 0),
        close_auction_end_time=get_ts(date, 15, 0, 0),  # match at 15:00, all close call auction orders are in [14:57, 15, 00)
        continuous_auction_start_time=get_ts(date, 9, 30, 0),
        continuous_auction_end_time=get_ts(date, 14, 56, 59, 999999),
    )
    return config


def create_exchange_config_without_call_auction(market_open: Timestamp, market_close: Timestamp, symbols: list[str]) -> ExchangeConfig:
    """Create exchange configuration without call auction."""
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
