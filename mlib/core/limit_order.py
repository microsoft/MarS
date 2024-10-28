# pyright: strict
from typing import TYPE_CHECKING, List

import pandas as pd

from mlib.core.base_order import BaseOrder

if TYPE_CHECKING:
    from mlib.core.orderbook import Orderbook


class LimitOrder(BaseOrder):
    """Currently, just make LimitOrder immutable from outside of this class.
    In the future, if we observe significant performance overhead,
    it should be easy to make them public mutable.
    """

    def __init__(
        self,
        time: pd.Timestamp,
        type: str,
        price: int,
        volume: int,
        symbol: str,
        agent_id: int,
        order_id: int,
        cancel_type: str,
        cancel_id: int,
        tag: str,
    ) -> None:
        super().__init__(symbol, time, agent_id, order_id)
        self._type = type
        self._price = price
        self._volume = volume
        self._cancel_type = cancel_type
        self._cancel_id = cancel_id
        self._tag = tag

    def __repr__(self) -> str:
        return f"order id: {self.order_id}, price {self.price}, volume {self.volume}, type: {self.type}, cancel: {self.is_cancel}({self.cancel_type})"

    @property
    def type(self):
        return self._type

    @property
    def price(self):
        return self._price

    @property
    def volume(self):
        return self._volume

    @property
    def cancel_type(self):
        return self._cancel_type

    @property
    def cancel_id(self):
        return self._cancel_id

    @property
    def tag(self):
        return self._tag

    @property
    def is_cancel(self):
        return self._type == "C"

    @property
    def is_buy(self):
        return self._type == "B"

    @property
    def is_sell(self):
        return self._type == "S"

    @property
    def is_cancel_buy(self):
        return self.is_cancel and self.cancel_type == "B"

    @property
    def is_cancel_sell(self):
        return self.is_cancel and self.cancel_type == "S"

    def decrease_volume(self, volume_to_decrease: int):
        assert volume_to_decrease > 0
        self._volume -= volume_to_decrease

    def clone(self):
        return LimitOrder(
            time=self.time,
            type=self.type,
            price=self.price,
            volume=self.volume,
            symbol=self.symbol,
            agent_id=self.agent_id,
            order_id=self.order_id,
            cancel_type=self.cancel_type,
            cancel_id=self.cancel_id,
            tag=self.tag,
        )

    def get_limit_orders(self, orderbook: "Orderbook") -> List["LimitOrder"]:
        return [self.clone()]
