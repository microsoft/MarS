import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pandas import Timedelta, Timestamp

from market_simulation.states.order_state import PredOrderInfo
from market_simulation.states.trade_info_state import TradeInfoState
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent

# from mlib.core.time_utils import get_ts
from mlib.core.limit_order import LimitOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.observation import Observation
from mlib.core.transaction import Transaction

TradingStage = Literal[
    "passive",
    "aggressive",
]


class TradingAgent(BaseAgent):
    """A agent that trade like Twap."""

    def __init__(
        self,
        symbol: str,
        start_time: Timestamp,
        total_seconds: int,
        direction: Literal["B", "S"],
        max_passive_volume_ratio: float = 0.1,
        max_aggressive_volume_ratio: float = 1,
        target_volume: int = 1000,
        passive_seconds: int = 20,
        idle_seconds: int = 10,
        aggressive_price_change: int = 500,
    ) -> None:
        super().__init__(init_cash=0, communication_delay=0, computation_delay=0)
        self.symbol = symbol
        self.passive_seconds = passive_seconds
        self.idle_seconds = idle_seconds
        self.start_time = start_time
        self.total_seconds = total_seconds
        self.end_time = start_time + Timedelta(seconds=total_seconds)
        self.direction = direction
        self.target_volume = target_volume
        assert target_volume >= 100
        assert target_volume % 100 == 0
        self.stage: TradingStage = "passive"
        self.completed_volume = 0
        self.aggressive_price_change = aggressive_price_change
        self.max_passive_volume_ratio = max_passive_volume_ratio
        self.max_aggressive_volume_ratio = max_aggressive_volume_ratio
        # print agent parameters
        print(
            f"TradingAgent {self.start_time=} {self.direction=} {self.target_volume=} {self.aggressive_price_change=} {max_passive_volume_ratio=} {max_aggressive_volume_ratio=}"
        )

    def get_trivial_action(self, observation: Observation) -> Optional[Action]:
        cur_time = observation.time
        if cur_time >= self.end_time:
            lob_orders = list(self.lob_orders[self.symbol].values())
            cancel_orders = get_cancel_orders(lob_orders)
            limit_orders = self.get_limit_orders(cur_time, cancel_orders)
            if lob_orders:
                print(
                    f"{cur_time=}, Cancel all remaining volume {self.direction=} {self.completed_volume =} of {self.target_volume=}, {limit_orders=}."
                )
            return Action(
                agent_id=self.agent_id,
                orders=limit_orders,  # type: ignore
                time=cur_time,
                next_wakeup_time=None,
            )

        if cur_time < self.start_time:
            # submit nothing and wakeup
            return Action(
                agent_id=self.agent_id,
                orders=[],
                time=cur_time,
                next_wakeup_time=self.start_time,
            )
        return None

    def get_action(self, observation: Observation) -> Action:
        trivial_action = self.get_trivial_action(observation)
        if trivial_action is not None:
            return trivial_action

        if self.completed_volume >= self.target_volume:
            print(f"Finished {self.direction=} {self.completed_volume =} of {self.target_volume=}.")
            return Action(
                agent_id=self.agent_id,
                orders=[],
                time=observation.time,
                next_wakeup_time=None,
            )

        cur_time = observation.time
        elapsed_seconds = int(np.round((cur_time - self.start_time).total_seconds()))
        lob = self.get_lob()

        if self.direction == "B":
            orders, sleep_seconds = get_buy_action(
                stage=self.stage,
                target_volume=self.target_volume,
                completed_volume=self.completed_volume,
                lob_orders=list(self.lob_orders[self.symbol].values()),
                lob_snapshot=lob,
                max_passive_volume_ratio=self.max_passive_volume_ratio,
                max_aggressive_volume_ratio=self.max_aggressive_volume_ratio,
                aggressive_price_change=self.aggressive_price_change,
                elapsed_seconds=elapsed_seconds,
                total_seconds=self.total_seconds,
                passive_seconds=self.passive_seconds,
                idle_seconds=self.idle_seconds,
            )
        else:
            orders, sleep_seconds = get_sell_action(
                stage=self.stage,
                target_volume=self.target_volume,
                completed_volume=self.completed_volume,
                lob_orders=list(self.lob_orders[self.symbol].values()),
                lob_snapshot=lob,
                max_passive_volume_ratio=self.max_passive_volume_ratio,
                max_aggressive_volume_ratio=self.max_aggressive_volume_ratio,
                aggressive_price_change=self.aggressive_price_change,
                elapsed_seconds=elapsed_seconds,
                total_seconds=self.total_seconds,
                passive_seconds=self.passive_seconds,
                idle_seconds=self.idle_seconds,
            )
        print(
            f"{cur_time=}, {self.stage=}, {self.target_volume=}, {self.completed_volume=}, {self.max_passive_volume_ratio=}, {self.max_aggressive_volume_ratio=}, {elapsed_seconds=} {orders=}, {sleep_seconds=}"
        )
        self.stage = "passive" if self.stage == "aggressive" else "aggressive"  # switch stage
        limit_orders = self.get_limit_orders(cur_time, orders)
        action = Action(
            agent_id=self.agent_id,
            orders=limit_orders,  # type: ignore
            time=cur_time,
            next_wakeup_time=cur_time + Timedelta(seconds=sleep_seconds),
        )
        return action

    def get_limit_orders(self, time: Timestamp, orders: List[PredOrderInfo]) -> List[LimitOrder]:
        limit_orders: List[LimitOrder] = []
        for order in orders:
            if order.volume == 0:
                continue
            if order.order_type in ["B", "S"]:
                limit_order = LimitOrder(
                    time=time,
                    type=order.order_type,
                    price=order.price,
                    volume=order.volume,
                    symbol=self.symbol,
                    agent_id=self.agent_id,
                    order_id=-1,
                    cancel_type="None",
                    cancel_id=-1,
                    tag="",
                )
                limit_orders.append(limit_order)
            else:
                price_orders = self.lob_price_orders[self.symbol]
                assert order.price in price_orders
                to_cancel_orders = list(price_orders[order.price].values())
                total_volume = sum(order.volume for order in to_cancel_orders)
                assert total_volume >= order.volume
                to_cancel_volume = order.volume
                for to_cancel_order in to_cancel_orders:
                    volume = min(to_cancel_order.volume, to_cancel_volume)
                    cancel_order = LimitOrder(
                        time=time,
                        type=order.order_type,
                        price=order.price,
                        volume=volume,
                        symbol=self.symbol,
                        agent_id=self.agent_id,
                        order_id=-1,
                        cancel_type=to_cancel_order.type,
                        cancel_id=to_cancel_order.order_id,
                        tag="",
                    )
                    limit_orders.append(cancel_order)
                    to_cancel_volume -= volume
                    if to_cancel_volume == 0:
                        break

        return limit_orders

    def get_lob(self):
        state = self.symbol_states[self.symbol][TradeInfoState.__name__]
        assert isinstance(state, TradeInfoState)
        assert state.trade_infos
        lob = state.trade_infos[-1].lob_snapshot
        return lob

    def on_order_executed(self, time: Timestamp, transaction: Transaction, trans_order_id_to_notify: int):
        order_id = trans_order_id_to_notify
        assert order_id in self.lob_orders[self.symbol]
        for id in transaction.buy_id + transaction.sell_id:
            if id != order_id or transaction.type == "C":
                continue
            trans_volume = transaction.volume if transaction.order_matched_volume is None else transaction.order_matched_volume[id]
            self.completed_volume += trans_volume
        return super().on_order_executed(time, transaction, trans_order_id_to_notify)


def get_100_padded_volume(volume: int) -> int:
    """Get volume padded by 100.

    Example: 100 -> 100, 101 -> 200, 199 -> 200, 200 -> 200, 201 -> 300.
    """
    if volume % 100 == 0:
        return volume
    volume = (volume // 100 + 1) * 100
    return volume


def get_cancel_orders(orders_to_cancel: List[LimitOrder], total_cancel_volume: Optional[int] = None) -> List[PredOrderInfo]:
    """Create orders to cancel orders_to_cancel.

    First, it aggregates orders by price, then create cancel orders for each price level.
    """
    cancel_orders: List[PredOrderInfo] = []
    price_volumes: Dict[int, int] = {}
    if total_cancel_volume is None:
        total_cancel_volume = sum(order.volume for order in orders_to_cancel)
    assert total_cancel_volume is not None
    for order in orders_to_cancel:
        volume = min(order.volume, total_cancel_volume)
        price_volumes[order.price] = price_volumes.get(order.price, 0) + volume
        total_cancel_volume -= volume
        assert total_cancel_volume is not None
        assert total_cancel_volume >= 0
        if total_cancel_volume == 0:
            break

    for price, volume in price_volumes.items():
        cancel_orders.append(PredOrderInfo(order_type="C", price=price, volume=volume, interval=0))

    return cancel_orders


def get_ask1_bid1_prices(lob_snapshot: LobSnapshot) -> Tuple[int, int]:
    """Get ask1 price and bid1 price from lob_snapshot."""
    ask1_price = lob_snapshot.last_price
    if lob_snapshot.ask_prices:
        ask1_price = lob_snapshot.ask_prices[0]
    elif lob_snapshot.bid_prices:
        ask1_price = lob_snapshot.bid_prices[0] + 100

    bid1_price = lob_snapshot.last_price
    if lob_snapshot.bid_prices:
        bid1_price = lob_snapshot.bid_prices[0]
    elif lob_snapshot.ask_prices:
        bid1_price = lob_snapshot.ask_prices[0] - 100

    return ask1_price, bid1_price


def get_buy_action(
    stage: TradingStage,
    target_volume: int,
    completed_volume: int,
    lob_orders: List[LimitOrder],
    lob_snapshot: LobSnapshot,
    aggressive_price_change: int,
    max_passive_volume_ratio: float,
    max_aggressive_volume_ratio: float,
    elapsed_seconds: int,
    total_seconds: int = 300,
    passive_seconds: int = 20,
    idle_seconds: int = 10,
) -> Tuple[List[PredOrderInfo], int]:
    """Get buy action.

    The basic idea is to split the trading time into several (passive_seconds + idle_seconds), each with similar volume.

    Passive stage:
    1. cancel all orders whose price is not the bid1_price (best bid price)
    2. add passive orders to reach partial_volume
    3. sleep passive_seconds

    Aggressive stage:
    1. calculate expected_completed_volume based on total_seconds, elapsed_seconds, and target_volume
    2. if expected_completed_volume <= completed_volume, do nothing and sleep idle_seconds
    3. if available_volume < new_volume, cancel all passive orders and buy all remaining;
       else add aggressive orders to reach expected_completed_volume, sleep idle_seconds

    Note:
    - all buy/sell orders must be padded by 100, except for the last order.
    - passive order: buy at bid1_price, sell at ask1_price
    - aggressive order: buy at ask1_price + aggressive_price_change, sell at bid1_price - aggressive_price_change
    """
    assert total_seconds > passive_seconds + idle_seconds
    assert total_seconds % (passive_seconds + idle_seconds) == 0
    remaining_volume = target_volume - completed_volume
    max_passive_volume = get_100_padded_volume(int(target_volume * max_passive_volume_ratio))
    max_aggressive_volume = get_100_padded_volume(int(target_volume * max_aggressive_volume_ratio))

    to_submit_orders: List[PredOrderInfo] = []

    ask1_price, bid1_price = get_ask1_bid1_prices(lob_snapshot)
    if stage == "passive":
        # cancel all orders whose price is not the bid1_price (best bid price)
        to_cancel_orders: List[LimitOrder] = [order for order in lob_orders if order.price != bid1_price]  # to submit
        lob_orders = [order for order in lob_orders if order.price == bid1_price]
        assert all(order.type == "B" for order in to_cancel_orders)
        to_submit_orders.extend(get_cancel_orders(to_cancel_orders))

        # submit passive orders
        existing_bid1_volume = sum(order.volume for order in lob_orders)
        available_volume = remaining_volume - existing_bid1_volume
        # assert existing_bid1_volume <= remaining_volume
        if existing_bid1_volume < max_passive_volume:
            new_bid1_volume = get_100_padded_volume(max_passive_volume - existing_bid1_volume)
            new_bid1_volume = min(new_bid1_volume, available_volume)
            assert new_bid1_volume + existing_bid1_volume <= remaining_volume, f"{new_bid1_volume=}, {existing_bid1_volume=}, {remaining_volume=}"
            assert new_bid1_volume >= 0
            to_submit_orders.append(PredOrderInfo(order_type="B", price=bid1_price, volume=new_bid1_volume, interval=0))
        elif existing_bid1_volume - max_passive_volume >= 100:
            cancel_volume = existing_bid1_volume - max_passive_volume
            to_submit_orders.extend(get_cancel_orders(lob_orders, cancel_volume))
        return to_submit_orders, passive_seconds
    else:
        assert stage == "aggressive"
        assert elapsed_seconds + idle_seconds <= total_seconds
        expected_completed_volume = np.round(target_volume * (elapsed_seconds + idle_seconds) / total_seconds)
        if expected_completed_volume <= completed_volume:
            return [], idle_seconds
        new_volume = get_100_padded_volume(int(expected_completed_volume - completed_volume))
        new_volume = min(new_volume, max_aggressive_volume)
        available_volume = remaining_volume - sum(order.volume for order in lob_orders)
        if available_volume < new_volume:
            #  cancel all passive orders and sell all remaining if available_volume < new_volume
            to_submit_orders.extend(get_cancel_orders(lob_orders, new_volume - available_volume))
            available_volume += sum(order.volume for order in to_submit_orders)
            assert available_volume + 100 > new_volume, f"{available_volume=}, {new_volume=}"
            new_volume = available_volume
        assert 0 <= new_volume <= available_volume <= remaining_volume
        if new_volume > 0:
            to_submit_orders.append(
                PredOrderInfo(order_type="B", price=ask1_price + aggressive_price_change, volume=new_volume, interval=0)
            )  # buy at market price
        return to_submit_orders, idle_seconds


def get_sell_action(
    stage: TradingStage,
    target_volume: int,
    completed_volume: int,
    lob_orders: List[LimitOrder],
    lob_snapshot: LobSnapshot,
    aggressive_price_change: int,
    max_passive_volume_ratio: float,
    max_aggressive_volume_ratio: float,
    elapsed_seconds: int,
    total_seconds: int,
    passive_seconds: int = 20,
    idle_seconds: int = 10,
):
    """Get sell action."""
    assert total_seconds > passive_seconds + idle_seconds
    assert total_seconds % (passive_seconds + idle_seconds) == 0
    remaining_volume = target_volume - completed_volume
    max_passive_volume = get_100_padded_volume(int(target_volume * max_passive_volume_ratio))
    max_aggressive_volume = get_100_padded_volume(int(target_volume * max_aggressive_volume_ratio))
    to_submit_orders: List[PredOrderInfo] = []

    ask1_price, bid1_price = get_ask1_bid1_prices(lob_snapshot)
    if stage == "passive":
        # cancel all orders whose price is not the ask1_price (best ask price)
        to_cancel_orders: List[LimitOrder] = [order for order in lob_orders if order.price != ask1_price]  # to submit
        lob_orders = [order for order in lob_orders if order.price == ask1_price]
        assert all(order.type == "S" for order in to_cancel_orders)
        to_submit_orders.extend(get_cancel_orders(to_cancel_orders))

        # submit passive orders
        existing_ask1_volume = sum(order.volume for order in lob_orders)
        available_volume = remaining_volume - existing_ask1_volume
        # assert existing_ask1_volume <= remaining_volume
        if existing_ask1_volume < max_passive_volume:
            new_ask1_volume = get_100_padded_volume(max_passive_volume - existing_ask1_volume)
            new_ask1_volume = min(new_ask1_volume, available_volume)
            assert new_ask1_volume + existing_ask1_volume <= remaining_volume, f"{new_ask1_volume=}, {existing_ask1_volume=}, {remaining_volume=}"
            to_submit_orders.append(PredOrderInfo(order_type="S", price=ask1_price, volume=new_ask1_volume, interval=0))
        elif existing_ask1_volume - max_passive_volume >= 100:
            cancel_volume = existing_ask1_volume - max_passive_volume
            to_submit_orders.extend(get_cancel_orders(lob_orders, cancel_volume))
        return to_submit_orders, passive_seconds
    else:
        assert stage == "aggressive"
        assert elapsed_seconds + idle_seconds <= total_seconds
        expected_completed_volume = np.round(target_volume * (elapsed_seconds + idle_seconds) / total_seconds)
        if expected_completed_volume <= completed_volume:
            return [], idle_seconds
        new_volume = get_100_padded_volume(int(expected_completed_volume - completed_volume))
        new_volume = min(new_volume, max_aggressive_volume)
        available_volume = remaining_volume - sum(order.volume for order in lob_orders)
        if available_volume < new_volume:
            #  cancel all passive orders and sell all remaining if available_volume < new_volume
            to_submit_orders.extend(get_cancel_orders(lob_orders, new_volume - available_volume))
            available_volume += sum(order.volume for order in to_submit_orders)
            assert available_volume + 100 > new_volume, f"{available_volume=}, {new_volume=}"
            new_volume = available_volume
        assert 0 <= new_volume <= available_volume <= remaining_volume
        if new_volume > 0:
            to_submit_orders.append(
                PredOrderInfo(order_type="S", price=bid1_price - aggressive_price_change, volume=new_volume, interval=0)
            )  # sell at market price
        return to_submit_orders, idle_seconds


def _get_limit_order(type: str, price: int, volume: int) -> LimitOrder:
    return LimitOrder(
        time=Timestamp("2021-01-01 09:30:00"),
        type=type,
        price=price,
        volume=volume,
        symbol="TEST",
        agent_id=1,
        order_id=-1,
        cancel_type="B",
        cancel_id=-1,
        tag="",
    )


def _test_get_buy_action() -> None:
    # case 1: submit passive orders
    lob_snapshot = LobSnapshot(
        time=Timestamp("2021-01-01 09:30:00"),
        last_price=12500,
        bid_prices=[12500, 12400, 12300, 12200],
        ask_prices=[12600, 12700, 12800, 12900],
        max_level=10,
        ask_volumes=[350, 1000, 650, 500],
        bid_volumes=[500, 650, 100, 350],
    )
    orders, sleep_seconds = get_buy_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[],
        lob_snapshot=lob_snapshot,
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1 and orders[0].price == 12500 and orders[0].volume == 100
    assert sleep_seconds == 20

    orders, sleep_seconds = get_buy_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[],
        lob_snapshot=lob_snapshot,
        max_passive_volume_ratio=0.4,
        max_aggressive_volume_ratio=1,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1 and orders[0].price == 12500 and orders[0].volume == 200
    assert sleep_seconds == 20

    # cancel all orders that are not bid1_price, and then submit passive orders
    orders, sleep_seconds = get_buy_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("B", 12400, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert orders[0].price == 12400 and orders[0].volume == 100 and orders[0].order_type == "C"
    assert orders[1].price == 12500 and orders[1].volume == 100 and orders[1].order_type == "B"
    assert sleep_seconds == 20

    orders, sleep_seconds = get_buy_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("B", 12400, 100),
        ],
        max_passive_volume_ratio=0.4,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert orders[0].price == 12400 and orders[0].volume == 100 and orders[0].order_type == "C"
    assert orders[1].price == 12500 and orders[1].volume == 200 and orders[1].order_type == "B"
    assert sleep_seconds == 20

    # submit aggressive orders
    orders, sleep_seconds = get_buy_action(
        stage="aggressive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("B", 12500, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1 and orders[0].price == 13600 and orders[0].volume == 100 and orders[0].order_type == "B"
    assert sleep_seconds == 10

    # cancel passive orders and buy needed volume if available_volume < new_volume
    orders, sleep_seconds = get_buy_action(
        stage="aggressive",
        target_volume=150,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("B", 12500, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 2
    assert orders[0].price == 12500 and orders[0].volume == 50 and orders[0].order_type == "C"
    assert orders[1].price == 13600 and orders[1].volume == 100 and orders[1].order_type == "B"
    assert sleep_seconds == 10

    # submit more volumes if completed_volume << expected_completed_volume
    orders, sleep_seconds = get_buy_action(
        stage="aggressive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("B", 12500, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=100,  # 110 + 10 == 120, 2/5 of total time
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1
    assert orders[0].price == 13600 and orders[0].volume == 200 and orders[0].order_type == "B", f"{orders=}"
    assert sleep_seconds == 10


def _test_get_sell_action() -> None:
    # case 1: submit passive orders
    lob_snapshot = LobSnapshot(
        time=Timestamp("2021-01-01 09:30:00"),
        last_price=12500,
        bid_prices=[12500, 12400, 12300, 12200],
        ask_prices=[12600, 12700, 12800, 12900],
        max_level=10,
        ask_volumes=[350, 1000, 650, 500],
        bid_volumes=[500, 650, 100, 350],
    )
    orders, sleep_seconds = get_sell_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1 and orders[0].price == 12600 and orders[0].volume == 100

    orders, sleep_seconds = get_sell_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[],
        max_passive_volume_ratio=0.4,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1 and orders[0].price == 12600 and orders[0].volume == 200

    # cancel all orders that are not ask1_price, and then submit passive orders
    orders, sleep_seconds = get_sell_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("S", 12700, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert orders[0].price == 12700 and orders[0].volume == 100 and orders[0].order_type == "C"
    assert orders[1].price == 12600 and orders[1].volume == 100 and orders[1].order_type == "S"
    assert sleep_seconds == 20

    orders, sleep_seconds = get_sell_action(
        stage="passive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("S", 12700, 100),
        ],
        max_passive_volume_ratio=0.4,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert orders[0].price == 12700 and orders[0].volume == 100 and orders[0].order_type == "C"
    assert orders[1].price == 12600 and orders[1].volume == 200 and orders[1].order_type == "S"
    assert sleep_seconds == 20

    # submit aggressive orders
    orders, sleep_seconds = get_sell_action(
        stage="aggressive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("S", 12600, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1 and orders[0].price == 11500 and orders[0].volume == 100 and orders[0].order_type == "S"
    assert sleep_seconds == 10

    # cancel all passive orders and sell all remaining if available_volume < new_volume
    orders, sleep_seconds = get_sell_action(
        stage="aggressive",
        target_volume=150,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("S", 12600, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=0,
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 2
    assert orders[0].price == 12600 and orders[0].volume == 50 and orders[0].order_type == "C"
    assert orders[1].price == 11500 and orders[1].volume == 100 and orders[1].order_type == "S"
    assert sleep_seconds == 10

    # submit more volumes if completed_volume << expected_completed_volume
    orders, sleep_seconds = get_sell_action(
        stage="aggressive",
        target_volume=500,
        completed_volume=0,
        lob_orders=[
            _get_limit_order("S", 12600, 100),
        ],
        max_passive_volume_ratio=0.1,
        max_aggressive_volume_ratio=1,
        lob_snapshot=lob_snapshot,
        aggressive_price_change=1000,
        elapsed_seconds=100,  # 110 + 10 == 120, 2/5 of total time
        total_seconds=300,
        passive_seconds=20,
        idle_seconds=10,
    )
    assert len(orders) == 1
    assert orders[0].price == 11500 and orders[0].volume == 200 and orders[0].order_type == "S", f"{orders=}"
    assert sleep_seconds == 10


def _test() -> None:
    _test_get_buy_action()
    _test_get_sell_action()


if __name__ == "__main__":
    _test()
    logging.info("Done!")
