# pyright: strict

from mlib.core.limit_order import LimitOrder
from mlib.core.transaction import Transaction


class Level:
    """LOB level class."""

    def __init__(self, price: int, volume: int, orders: list[LimitOrder]) -> None:
        self.price: int = price
        self._volume: int = volume
        self._orders: dict[int, LimitOrder] = {x.order_id: x for x in orders}
        self.check()

    @property
    def volume(self) -> int:
        """Get the volume of the level."""
        return self._volume

    @property
    def orders(self) -> list[LimitOrder]:
        """Get the orders of the level."""
        return list(self._orders.values())

    def has_order_id(self, id: int) -> bool:
        """Check if the level has the order id."""
        return id in self._orders

    def update_with_cancel_order(self, cancel_order: LimitOrder) -> Transaction:
        """Update the level with a cancel order."""
        assert cancel_order.is_cancel
        self._volume -= cancel_order.volume
        index = cancel_order.cancel_id
        assert index in self._orders
        self._orders[index].decrease_volume(cancel_order.volume)
        trans_info = Transaction(
            symbol=cancel_order.symbol,
            time=cancel_order.time,
            type="C",
            price=cancel_order.price,
            volume=cancel_order.volume,
            buy_id=[self._orders[index].order_id] if self._orders[index].is_buy else [],
            sell_id=[self._orders[index].order_id] if self._orders[index].is_sell else [],
        )
        assert self._orders[index].volume >= 0
        if self._orders[index].volume == 0:
            del self._orders[index]
        self.check()
        return trans_info

    def add_new_order(self, new_order: LimitOrder) -> None:
        """Add a new order to the level."""
        self._volume += new_order.volume
        self._orders[new_order.order_id] = new_order
        assert new_order.price == self.price
        assert new_order.volume > 0

    def update_with_clear_order(self, clear_order: LimitOrder) -> tuple[LimitOrder, int, list[tuple[int, int]]]:
        """Update the level with a clear order."""
        assert clear_order.is_buy or clear_order.is_sell
        matched_volume = 0
        matched_details: list[tuple[int, int]] = []
        for order in self._orders.values():
            assert order.type != clear_order.type
            vol = min(order.volume, clear_order.volume)
            order.decrease_volume(vol)
            clear_order.decrease_volume(vol)
            matched_volume += vol
            matched_details.append((order.order_id, vol))
            self._volume -= vol
            if clear_order.volume == 0:
                break
        cleared_order_ids = [x.order_id for x in self._orders.values() if x.volume == 0]
        for order_id in cleared_order_ids:
            del self._orders[order_id]
        self.check()
        return clear_order, matched_volume, matched_details

    def check(self) -> None:
        """Check the level."""
        for order in self._orders.values():
            assert self.price == order.price
            assert self.volume > 0
        # check total volume
        if not self._orders:
            assert self._volume == 0
        else:
            assert self._volume == sum([x.volume for x in self._orders.values()])

    def __repr__(self) -> str:
        return f"price {self.price}, volume {self._volume}"
