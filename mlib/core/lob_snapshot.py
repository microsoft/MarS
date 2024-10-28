# pyright: strict
import logging
from typing import List, NamedTuple

import numpy as np
import pandas as pd


class LobSnapshot(NamedTuple):
    """Snapshot of limit orderbook."""

    time: pd.Timestamp
    max_level: int
    last_price: int
    ask_prices: List[int]
    ask_volumes: List[int]
    bid_prices: List[int]
    bid_volumes: List[int]

    @property
    def spread(self) -> int:
        if self.ask_prices and self.bid_prices:
            spread = self.ask_prices[0] - self.bid_prices[0]
        else:
            real_price_interval = 100
            spread = real_price_interval * 10
        return spread

    @property
    def mid_price(self):
        """Get mid price of LOB snapshot, the returned price is a valid tick on LOB."""
        spread: int = self.spread
        real_price_interval = 100
        bid0 = self.bid_prices[0] if self.bid_prices else self.last_price - real_price_interval * 10
        if self.ask_prices and self.bid_prices:
            return bid0 + spread // real_price_interval // 2 * real_price_interval
        elif self.ask_prices:
            logging.debug("bid is empty, return ask0 for mid price")
            return self.ask_prices[0]
        elif self.bid_prices:
            logging.debug("ask is empty, return bid0 for mid price")
            return self.bid_prices[0]
        elif self.last_price > 0:
            logging.debug(f"Both ask/bid are empty, return cur price: {self.last_price}")
            return self.last_price
        else:
            raise ValueError("unknown mid price")

    @property
    def float_mid_price(self) -> float:
        """Get mid price of LOB snapshot, the returned price is a float value and may be not valid on LOB."""
        if not self.ask_prices and not self.bid_prices:
            print(f"warning: empty LOB (no ask/bid prices), last price: {self.last_price}")
            assert self.last_price is not None
            assert not np.isnan(self.last_price)
            return self.last_price
        if not self.ask_prices:
            return self.bid_prices[0]
        if not self.bid_prices:
            return self.ask_prices[0]
        assert self.bid_volumes[0] > 0
        assert self.ask_volumes[0] > 0
        mid_price = (self.bid_prices[0] + self.ask_prices[0]) / 2
        assert self.bid_prices[0] < mid_price < self.ask_prices[0]
        return mid_price

    @property
    def float_weighted_mid_price(self) -> float:
        """Get weighted-mid price of LOB snapshot."""
        if not self.ask_prices and not self.bid_prices:
            print(f"warning: empty LOB (no ask/bid prices), last price: {self.last_price}")
            assert self.last_price is not None
            assert not np.isnan(self.last_price)
            return self.last_price
        if not self.ask_prices:
            return self.bid_prices[0]
        if not self.bid_prices:
            return self.ask_prices[0]
        assert self.bid_volumes[0] > 0
        assert self.ask_volumes[0] > 0
        weighted_mid_price = (self.ask_prices[0] * self.bid_volumes[0] + self.bid_prices[0] * self.ask_volumes[0]) / (
            self.bid_volumes[0] + self.ask_volumes[0]
        )
        assert self.bid_prices[0] < weighted_mid_price < self.ask_prices[0]
        return weighted_mid_price
