# pyright: strict

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from pandas import Timestamp

# fix circulate import caused by type hints: https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from mlib.core.limit_order import LimitOrder
    from mlib.core.orderbook import Orderbook


class BaseOrder(ABC):
    """The base class for order.

    A base order should implement `get_limit_orders` that convert the base order to limit orders with orderbook information.
    """

    def __init__(self, symbol: str, time: Timestamp, agent_id: int = -1, order_id: int = -1):
        self._symbol = symbol
        self._time = time
        self._agent_id = agent_id
        self._order_id = order_id

    @property
    def time(self):
        return self._time

    @property
    def symbol(self):
        return self._symbol

    @property
    def agent_id(self):
        return self._agent_id

    def set_agent_id(self, agent_id: int):
        self._agent_id = agent_id

    @property
    def order_id(self):
        return self._order_id

    def set_order_id(self, order_id: int):
        assert self._order_id < 0
        self._order_id = order_id

    @abstractmethod
    def get_limit_orders(self, orderbook: "Orderbook") -> List["LimitOrder"]:
        """Conver to limit orders with orderbook information."""
        raise NotImplementedError()
