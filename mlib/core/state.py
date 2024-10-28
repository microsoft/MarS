# pyright: strict
from typing import List, Optional

from pandas import Timestamp

from mlib.core.exchange_config import ExchangeConfig
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.orderbook import Orderbook
from mlib.core.trade_info import TradeInfo
from mlib.core.transaction import Transaction


class State:
    """The base class for state."""

    def __init__(self) -> None:
        self.open_price = -1
        self.open_volume = -1
        self.close_price = -1
        self.close_volume = -1
        self.config: ExchangeConfig
        self.time: Timestamp = Timestamp.now()
        self.last_price = -1
        self.lob_snapshot: Optional[LobSnapshot] = None
        self.close_orderbook: Optional[Orderbook] = None

    def on_trading(self, trade_info: TradeInfo):
        """Update with trade info during continuous auction period, which includes a limit order, orderbook snapshot and transactions."""
        self.time = trade_info.order.time
        self.lob_snapshot = trade_info.lob_snapshot
        for trans in trade_info.transactions:
            if trans.type in ["B", "S"]:
                self.last_price = trans.price

    def on_call_auction_trading(self, trade_info: TradeInfo):
        """Update with trade info during the call auction period.

        No transaction is available in call auction period.
        """
        pass

    def on_open(
        self,
        cancel_transactions: List[Transaction],
        lob_snapshot: LobSnapshot,
        match_trans: Optional[Transaction] = None,
    ):
        """Update with trade info from the final order matching of open call auction."""
        if match_trans:
            self.open_price = match_trans.price
            self.open_volume = match_trans.volume
        self.lob_snapshot = lob_snapshot

    def on_close(
        self,
        close_orderbook: Orderbook,
        lob_snapshot: LobSnapshot,
        match_trans: Optional[Transaction] = None,
    ):
        """Update with trade info from the final order matching of close call auction."""
        if match_trans:
            self.close_price = match_trans.price
            self.close_volume = match_trans.volume
        self.close_orderbook = close_orderbook
        self.lob_snapshot = lob_snapshot
