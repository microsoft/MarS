# pyright: strict, reportUnknownMemberType=false
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Deque, List, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_simulation.utils import pkl_utils
from mlib.core.limit_order import LimitOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.state import State
from mlib.core.trade_info import TradeInfo
from mlib.core.transaction import Transaction

if TYPE_CHECKING:
    from market_simulation.utils.bin_converter import BinConverter


class Converter:
    """A collection of converters."""

    def __init__(self, converter_dir: Path) -> None:
        self.price: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "price.zstd")
        self.price_level: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "price-level.zstd")
        self.price_change_ratio: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "price-change-ratio.zstd")
        self.order_volume: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "order-volume.zstd")
        self.lob_volume: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "lob-volume.zstd")
        self.pred_order_volume: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "pred-order-volume.zstd")
        self.order_interval: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "order-interval.zstd")
        self.minute_buy_order_count: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "minute-buy-order-count.zstd")
        self.minute_trans_vwap_change: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "minute-trans-vwap-change.zstd")
        self.minute_trans_volume: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "minute-trans-volume.zstd")
        self.num_minutes_to_open: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "num-minutes-to-open.zstd")
        self.minute_cancel_volume: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "minute-cancel-volume.zstd")
        self.lob_spread: BinConverter = pkl_utils.load_pkl_zstd(converter_dir / "lob-spread.zstd")


class OrderInfo:
    """Order information."""

    NUM_RATIO_SLOTS: int = 10
    NUM_LOB_VOLUMES: int = 10

    def __init__(
        self,
        cur_order_lob: LobSnapshot,
        cur_order: LimitOrder,
        transactions: List[Transaction],
        order_index: int,
        interval_seconds: float,
        price_change_to_open: int,
        time_to_open: int,
        lob_volumes: List[int],
    ) -> None:
        self.time = cur_order.time
        self.trans_ratio = self.get_trans_ratio(transactions, cur_order)
        assert self.trans_ratio <= 1.00001
        self.volume_ratio = self.get_volume_ratio(cur_order, cur_order_lob)
        assert self.volume_ratio <= 1.00001
        self.order_index = order_index
        self.interval_seconds = interval_seconds
        self.price_change_to_open = price_change_to_open
        self.time_to_open = time_to_open
        self.lob_volumes = lob_volumes

    @staticmethod
    def get_trans_ratio(trans: List[Transaction], cur_order: LimitOrder) -> float:
        """Get transaction ratio, which is the ratio of transaction volume to order volume."""
        if cur_order.type == "C":
            return 1.0
        return sum([x.volume for x in trans]) / cur_order.volume

    def get_volume_ratio(self, cur_order: LimitOrder, cur_order_lob: LobSnapshot) -> float:
        """Get volume ratio, which is the ratio of order_volume / (lob_level_volume + order_volume)."""
        prices = cur_order_lob.ask_prices + cur_order_lob.bid_prices
        volumes = cur_order_lob.ask_volumes + cur_order_lob.bid_volumes
        assert len(prices) == len(volumes)
        if cur_order.price in prices:
            index = prices.index(cur_order.price)
            volume_ratio = (cur_order.volume) / (volumes[index] + cur_order.volume)
        else:
            volume_ratio = 1.0
        return volume_ratio

    def to_vector(self) -> npt.NDArray[np.int32]:
        """Convert order info to vector."""
        values: List[int] = [
            self.order_index,
            int(np.floor(self.volume_ratio * 0.99 * OrderInfo.NUM_RATIO_SLOTS)),  # volume ratio slot, [0, 9]
            int(np.floor(self.trans_ratio * 0.99 * OrderInfo.NUM_RATIO_SLOTS)),  # trans ratio slot, [0, 9],
            self.price_change_to_open,
            self.time_to_open,
            *self.lob_volumes,
        ]
        vec = np.array(values, dtype=np.int32)
        assert vec.size == self.get_dim()
        return vec

    @staticmethod
    def get_dim() -> int:
        """Get the dimension of order info."""
        return 5 + OrderInfo.NUM_LOB_VOLUMES


class PredOrderInfo(NamedTuple):
    """Pred order info."""

    order_type: str
    price: int
    volume: int
    interval: int

    @staticmethod
    def get_index_from_type(order_type: str) -> int:
        """Get index of order type."""
        return ["S", "B", "C"].index(order_type)

    @staticmethod
    def get_type_from_index(index: int) -> str:
        """Get order type from index."""
        return ["S", "B", "C"][index]


class OrderState(State):
    """Order state."""

    def __init__(
        self,
        num_max_orders: int,
        num_bins_price_level: int,
        num_bins_pred_order_volume: int,
        num_bins_order_interval: int,
        converter: Converter,
    ) -> None:
        super().__init__()
        self.num_max_orders = num_max_orders
        self.converter = converter
        self.recent_orders: Deque[OrderInfo] = deque()
        self.latest_lob: Optional[LobSnapshot] = None

        self.prev_order: Optional[LimitOrder] = None
        self.prev_order_lob: Optional[LobSnapshot] = None  # lob for prev_order
        self.cur_order_lob: Optional[LobSnapshot] = None
        self.cur_order: Optional[LimitOrder] = None
        self.open_trans_price: Optional[float] = None
        self.open_time: Optional[pd.Timestamp] = None
        self.num_bins_price_level = num_bins_price_level
        self.num_bins_pred_order_volume = num_bins_pred_order_volume
        self.num_bins_order_interval = num_bins_order_interval
        assert converter.price_level.num_bins == self.num_bins_price_level
        assert converter.pred_order_volume.num_bins == self.num_bins_pred_order_volume
        assert converter.order_interval.num_bins == self.num_bins_order_interval

    def on_trading(self, trade_info: TradeInfo) -> None:
        """Update order state with trading information."""
        if self.latest_lob is None:
            self.latest_lob = trade_info.lob_snapshot
            self.prev_order = trade_info.order
            self.open_time = trade_info.order.time
            return

        # set open price if need
        if self.open_trans_price is None and trade_info.transactions and trade_info.transactions[0].type in ["B", "S"]:
            self.open_trans_price = trade_info.transactions[0].price

        self.cur_order_lob = self.latest_lob
        self.latest_lob = trade_info.lob_snapshot
        self.cur_order = trade_info.order
        assert self.prev_order is not None
        self.update_order_info(trade_info)

    def get_seconds_to_open(self, cur_time: pd.Timestamp, open_time: pd.Timestamp) -> int:
        """Get seconds to market open.

        Note: currently this code is adapted to Chinese stock market, where we skip 11:30 to 13:00 when calculate the elapsed seconds.
        """
        assert cur_time >= open_time
        seconds = (cur_time - open_time).total_seconds()
        if cur_time.hour >= 13:
            seconds -= 1.5 * 3600  # empty from 11:30 to 13:00
        return int(seconds)

    def get_order_index_from_slots(
        self,
        order_type: int,
        price_slot: int,
        volume_slot: int,
        interval_slot: int,
    ) -> int:
        """Get order index from slots."""
        return (
            order_type * (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
            + price_slot * (self.num_bins_pred_order_volume * self.num_bins_order_interval)
            + volume_slot * self.num_bins_order_interval
            + interval_slot
        )

    def get_order_index(
        self,
        cur_order: LimitOrder,
        interval_seconds: float,
        cur_order_lob: LobSnapshot,
    ) -> int:
        """Get order index from order, interval and lob."""
        order_type = PredOrderInfo.get_index_from_type(cur_order.type)  # [0, 1, 2]
        price_slot = self.converter.price_level.get_bin_index(cur_order.price - cur_order_lob.mid_price)
        volume_slot = self.converter.order_volume.get_bin_index(cur_order.volume)
        interval_slot = self.converter.order_interval.get_bin_index(interval_seconds)
        return (
            order_type * (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
            + price_slot * (self.num_bins_pred_order_volume * self.num_bins_order_interval)
            + volume_slot * self.num_bins_order_interval
            + interval_slot
        )

    def get_pred_order_info(self, order_index: int) -> PredOrderInfo:
        """Reverse function of get_order_index, need a further sampling to get real price/volume/interval."""
        order_type = order_index // (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
        price_slot = (order_index % (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)) // (
            self.num_bins_pred_order_volume * self.num_bins_order_interval
        )
        volume_slot = (order_index % (self.num_bins_pred_order_volume * self.num_bins_order_interval)) // self.num_bins_order_interval
        interval_slot = order_index % self.num_bins_order_interval
        return PredOrderInfo(
            order_type=PredOrderInfo.get_type_from_index(order_type),
            price=price_slot,
            volume=volume_slot,
            interval=interval_slot,
        )

    def get_lob_volume_slots(self, lob: LobSnapshot, total_len: int = 10) -> List[int]:
        """Get volume slots from lob."""
        assert total_len % 2 == 0
        offset: int = total_len // 2
        volume_slots: List[int] = [0] * total_len
        mid_price = lob.mid_price
        for price, volume in zip(lob.ask_prices, lob.ask_volumes):
            price_slot = (price - mid_price) // 100 + offset
            if 0 <= price_slot < total_len:
                volume_slots[price_slot] = self.converter.lob_volume.get_bin_index(volume) + 1

        for price, volume in zip(lob.bid_prices, lob.bid_volumes):
            price_slot = (price - mid_price) // 100 + offset
            if 0 <= price_slot < total_len:
                volume_slots[price_slot] = self.converter.lob_volume.get_bin_index(volume) + 1
        return volume_slots

    def update_order_info(self, trade_info: TradeInfo) -> None:
        """Update order information."""
        assert self.cur_order is not None
        assert self.prev_order is not None
        assert self.cur_order_lob is not None
        assert self.open_time is not None
        mid_price: int = trade_info.lob_snapshot.mid_price
        price_change = 0 if self.open_trans_price is None else mid_price / self.open_trans_price - 1
        price_change = np.clip(price_change, -0.2, 0.2)
        seconds_to_open = self.get_seconds_to_open(self.cur_order.time, self.open_time)
        lob_volumes = self.get_lob_volume_slots(trade_info.lob_snapshot, total_len=OrderInfo.NUM_LOB_VOLUMES)
        # merge continous cancel orders if their prices and time are same.
        is_same_cancel_order: bool = (
            self.cur_order.type == "C"
            and self.prev_order.type == "C"
            and self.cur_order.time == self.prev_order.time
            and self.cur_order.price == self.prev_order.price
            and self.cur_order.symbol == self.prev_order.symbol
            and self.cur_order.agent_id == self.prev_order.agent_id
            and self.prev_order_lob is not None
        )

        if is_same_cancel_order:
            assert self.prev_order_lob is not None
            # merge cancel orders, and set prev_order with merged order..
            merged_order = self.prev_order.clone()
            merged_order._volume += self.cur_order.volume  # type: ignore  # noqa: SLF001
            last_order_info = self.recent_orders[-1]
            order_index = self.get_order_index(
                cur_order=merged_order,
                interval_seconds=last_order_info.interval_seconds,
                cur_order_lob=self.prev_order_lob,
            )
            order_info = OrderInfo(
                cur_order_lob=self.prev_order_lob,
                cur_order=merged_order,
                transactions=trade_info.transactions,
                interval_seconds=last_order_info.interval_seconds,
                order_index=order_index,
                price_change_to_open=int(price_change * 10000),  # [-2000, 2000]
                time_to_open=seconds_to_open,
                lob_volumes=lob_volumes,
            )

            self.recent_orders[-1] = order_info  # replace with mreged order info
            self.prev_order = merged_order
        else:
            interval_seconds = (trade_info.order.time - self.prev_order.time).total_seconds()
            order_index = self.get_order_index(
                cur_order=self.cur_order,
                interval_seconds=interval_seconds,
                cur_order_lob=self.cur_order_lob,
            )
            order_info = OrderInfo(
                cur_order_lob=self.cur_order_lob,
                cur_order=self.cur_order,
                transactions=trade_info.transactions,
                interval_seconds=(trade_info.order.time - self.prev_order.time).total_seconds(),
                order_index=order_index,
                price_change_to_open=int(price_change * 10000),
                time_to_open=seconds_to_open,
                lob_volumes=lob_volumes,
            )
            self.recent_orders.append(order_info)
            if self.num_max_orders > 0 and len(self.recent_orders) > self.num_max_orders:
                self.recent_orders.popleft()
            self.prev_order = self.cur_order
            self.prev_order_lob = self.cur_order_lob

    def to_vector(self) -> npt.NDArray[np.int32]:
        """Convert order state to vector."""
        vectors: List[npt.NDArray[np.int32]] = []

        assert self.cur_order is not None
        assert self.latest_lob is not None
        for i in range(len(self.recent_orders)):
            vectors.append(self.recent_orders[i].to_vector())
        state_vector: npt.NDArray[np.int32] = np.concatenate(vectors, dtype=np.int32)
        return state_vector


def _test_get_index() -> None:
    import logging

    from market_simulation.conf import C

    converter_dir = Path(C.directory.input_root_dir) / C.order_model.converter_dir
    converter = Converter(converter_dir)
    state = OrderState(
        num_max_orders=-1,
        num_bins_price_level=converter.price_level.num_bins,
        num_bins_pred_order_volume=converter.pred_order_volume.num_bins,
        num_bins_order_interval=converter.order_interval.num_bins,
        converter=converter,
    )

    index = 0
    for order_type in range(3):
        for price in range(32):
            for volume in range(32):
                for interval in range(16):
                    order_index = state.get_order_index_from_slots(order_type, price, volume, interval)
                    assert order_index == index
                    pred_order_info = state.get_pred_order_info(order_index)
                    type_i, price_i, volume_i, interval_i = pred_order_info
                    assert order_type == PredOrderInfo.get_index_from_type(type_i)
                    assert price == price_i
                    assert volume == volume_i
                    assert interval == interval_i
                    index += 1
    logging.info(f"Order index test passed, total {index} orders.")


if __name__ == "__main__":
    _test_get_index()
