from typing import List

from mlib.core.state import State
from mlib.core.trade_info import TradeInfo


class TradeInfoState(State):
    """A state contains all trade infos."""

    def __init__(self) -> None:
        super().__init__()
        self.trade_infos: List[TradeInfo] = []
        self.call_auction_trade_infos: List[TradeInfo] = []

    def on_trading(self, trade_info: TradeInfo):
        super().on_trading(trade_info)
        self.trade_infos.append(trade_info)

    def on_call_auction_trading(self, trade_info: TradeInfo):
        super().on_call_auction_trading(trade_info)
        self.call_auction_trade_infos.append(trade_info)
