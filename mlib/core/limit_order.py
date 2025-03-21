# pyright: strict
from typing import TYPE_CHECKING

import pandas as pd

from mlib.core.base_order import BaseOrder

if TYPE_CHECKING:
    from mlib.core.orderbook import Orderbook


class LimitOrder(BaseOrder):
    """Limit order."""

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
    def type(self) -> str:
        """Order type: B for buy, S for sell, C for cancel."""
        return self._type

    @property
    def price(self) -> int:
        """Order price."""
        return self._price

    @property
    def volume(self) -> int:
        """Order volume."""
        return self._volume

    @property
    def cancel_type(self) -> str:
        """Cancel type: B for buy, S for sell."""
        return self._cancel_type

    @property
    def cancel_id(self) -> int:
        """Order id of the order to be cancelled."""
        return self._cancel_id

    @property
    def tag(self) -> str:
        """Tag for the order."""
        return self._tag

    @property
    def is_cancel(self) -> bool:
        """Whether the order is a cancel order."""
        return self._type == "C"

    @property
    def is_buy(self) -> bool:
        """Whether the order is a buy order."""
        return self._type == "B"

    @property
    def is_sell(self) -> bool:
        """Whether the order is a sell order."""
        return self._type == "S"

    @property
    def is_cancel_buy(self) -> bool:
        """Whether the order is a cancel buy order."""
        return self.is_cancel and self.cancel_type == "B"

    @property
    def is_cancel_sell(self) -> bool:
        """Whether the order is a cancel sell order."""
        return self.is_cancel and self.cancel_type == "S"

    def decrease_volume(self, volume_to_decrease: int) -> None:
        """Decrease the volume of the order."""
        assert volume_to_decrease > 0
        self._volume -= volume_to_decrease

    def clone(self) -> "LimitOrder":
        """Clone the order."""
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

    def get_limit_orders(self, orderbook: "Orderbook") -> list["LimitOrder"]:
        """Convert to limit orders with orderbook information."""
        return [self.clone()]
