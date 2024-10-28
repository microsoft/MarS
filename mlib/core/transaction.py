# pyright: strict
from typing import Dict, List, NamedTuple, Optional

from pandas import Timestamp

TRANSACTION_TYPES = ["B", "S", "C", "OPEN", "CLOSE"]


class Transaction(NamedTuple):
    time: Timestamp
    symbol: str
    type: str
    price: int
    volume: int
    buy_id: List[int]
    sell_id: List[int]
    order_matched_volume: Optional[Dict[int, int]] = None

    def __str__(self) -> str:
        assert self.type in TRANSACTION_TYPES
        bid_id_list = self.buy_id
        ask_id_list = self.sell_id
        if len(self.buy_id) > 1:
            bid_id_set = set(self.buy_id)
            bid_id_list = list(bid_id_set)
            bid_id_list.sort()
        if len(self.sell_id) > 1:
            ask_id_set = set(self.sell_id)
            ask_id_list = list(ask_id_set)
            ask_id_list.sort()

        bid_ids = "\n\t".join([str(x) for x in bid_id_list])
        ask_ids = "\n\t".join([str(x) for x in ask_id_list])
        price_str = "0" if self.type == "C" else str(self.price)
        result = f"time {self.time}, symbol: {self.symbol}, type {self.type}, price {price_str}, vol {self.volume}, bid_ids {bid_ids}, ask_ids {ask_ids}"
        if self.order_matched_volume is not None:
            keys = sorted(self.order_matched_volume.keys())
            values = [self.order_matched_volume[key] for key in keys]
            volume_info = ", ".join([f"order volume: {key}: {value}, " for key, value in zip(keys, values)])
            result += volume_info
        return result
