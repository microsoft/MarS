# pyright: strict
import logging
import sys
from random import Random, shuffle
from typing import Dict, List, Optional, Tuple

from pandas import Timedelta, Timestamp

from mlib.core.action import Action
from mlib.core.base_order import BaseOrder
from mlib.core.limit_order import LimitOrder
from mlib.core.observation import Observation
from mlib.core.state import State
from mlib.core.transaction import Transaction


class BaseAgent:
    """The base class for agent.

    Agent's informations are automatically updated by env, including:
    - `symbol_states`: the latest available states to the current agent.
    - `lob_orders`: active orders on orderbook, organized by <symbol, order_id>.
    - `lob_price_orders`: active orders on orderbook, organized by <symbol, price, order_id>.
    - `holdings`: agent's holdings for symbols.
    - `cash`: agent's cash.
    - `tradable_holdings`: like `holdings`, but when the agent submits a sell order with volume `v` for symbol `s`,
        no matter if this sell order is matched or not, `tradable_holdings[s]` will be decreased with `v`.
    - `tradable_cash`: like `cash`, but when the agent submits a buy order with cash `c`,
        the `tradable_cash` will be decreased with `c`.
    """

    def __init__(
        self,
        init_cash: float = 0,
        communication_delay: int = 0,
        computation_delay: int = 0,
    ) -> None:
        self.agent_id: int = -1
        self.symbol_states: Dict[str, Dict[str, State]] = {}
        self.lob_orders: Dict[str, Dict[int, LimitOrder]] = {}  # symbol -> order_id -> order
        self.lob_price_orders: Dict[str, Dict[int, Dict[int, LimitOrder]]] = {}  # symbol -> price -> order_id -> order
        self.holdings: Dict[str, int] = {}
        self.cash: float = init_cash
        self.tradable_holdings: Dict[str, int] = {}
        self.tradable_cash: float = init_cash
        self.communication_delay = Timedelta(communication_delay, unit="second")
        self.computation_delay = Timedelta(computation_delay, unit="second")
        self.states_update_time = Timestamp("2000-01-01")

    def init_base_info(self, other: "BaseAgent") -> None:
        """Copy base information from other agent."""
        self.symbol_states = other.symbol_states
        self.lob_orders = other.lob_orders
        self.lob_price_orders = other.lob_price_orders
        self.holdings = other.holdings
        self.cash = other.cash
        self.tradable_holdings = other.tradable_holdings
        self.tradable_cash = other.tradable_cash
        self.communication_delay = other.communication_delay
        self.computation_delay = other.computation_delay
        self.states_update_time = other.states_update_time

    def get_action(self, observation: Observation) -> Action:
        """Get action given observation.

        It delegates its main functions to:
        - `get_next_wakeup_time` to get the next wakeup time, and
        - `get_orders` to get orders based on observation. `get_orders` will not be called for the first-time wakeup,
            when it's the market open wakeup.

        """
        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        # return empty order for the market open wakeup
        orders: List[BaseOrder] = [] if observation.is_market_open_wakup else self.get_orders(time)
        action = Action(
            agent_id=self.agent_id,
            time=time,
            orders=orders,
            next_wakeup_time=self.get_next_wakeup_time(time),
        )
        return action

    def get_orders(self, time: Timestamp) -> List[BaseOrder]:
        """Generate orders based on current known states, e.g., `self.symbol_states`, `self.holdings`, `self.cash`, etc."""
        raise NotImplementedError()

    def get_next_wakeup_time(self, time: Timestamp) -> Optional[Timestamp]:
        """Provide the next wakeup time."""
        raise NotImplementedError()

    def on_market_open(self, time: Timestamp, symbols: List[str]) -> None:
        self._init_holdings(symbols)

    def on_market_close(self, time: Timestamp) -> None:
        pass

    def _init_holdings(self, symbols: List[str]) -> None:
        for symbol in symbols:
            self.lob_orders[symbol] = {}
            self.lob_price_orders[symbol] = {}
            self.holdings[symbol] = 0
            self.tradable_holdings[symbol] = 0

    def on_order_accepted(self, time: Timestamp, orders: List[LimitOrder]) -> None:
        for order in orders:
            self._add_accepted_order(order)

    def on_order_rejected(self, time: Timestamp, base_orders: List[BaseOrder]) -> None:
        for base_order in base_orders:
            logging.info(f"order rejected: {base_order}: this is an invalid cancel order.")

    def on_order_ignored(self, time: Timestamp, base_orders: List[BaseOrder]) -> None:
        for base_order in base_orders:
            logging.info(f"order ignore: {base_order}, invalid time: {base_order.time}")

    def on_states_update(self, time: Timestamp, symbol_states: Dict[str, Dict[str, State]]) -> None:
        """Before agent wakeup, this function will be called automatically by the env to update agent with the latest available states."""
        self.symbol_states = symbol_states
        self.states_update_time = time

    def _add_accepted_order(self, order: LimitOrder) -> None:
        if order.is_cancel:
            return
        symbol = order.symbol
        assert symbol in self.lob_orders
        assert order.order_id not in self.lob_orders[symbol]
        self.lob_orders[symbol][order.order_id] = order
        if order.price not in self.lob_price_orders[symbol]:
            self.lob_price_orders[symbol][order.price] = {}
        self.lob_price_orders[symbol][order.price][order.order_id] = order
        if order.is_sell:
            self.tradable_holdings[symbol] -= order.volume
        if order.is_buy:
            self.tradable_cash -= order.volume * order.price

    def on_order_executed(self, time: Timestamp, transaction: Transaction, trans_order_id_to_notify: int):
        del_order_ids: List[Tuple[str, int]] = []
        assert transaction.volume > 0
        assert transaction.type in ["B", "S", "C", "OPEN", "CLOSE"]
        for buy_id in transaction.buy_id:
            if buy_id != trans_order_id_to_notify:
                continue
            assert buy_id in self.lob_orders[transaction.symbol]
            limit_order = self.lob_orders[transaction.symbol][buy_id]
            trans_volume: int = transaction.volume if transaction.order_matched_volume is None else transaction.order_matched_volume[buy_id]
            limit_order.decrease_volume(trans_volume)
            assert limit_order.volume >= 0
            if limit_order.volume == 0:
                self.lob_orders[limit_order.symbol].pop(limit_order.order_id)
                self.lob_price_orders[limit_order.symbol][limit_order.price].pop(limit_order.order_id)
                del_order_ids.append((limit_order.symbol, limit_order.order_id))
            if transaction.type != "C":
                # buy order was matched: cash turn into holding
                self.cash -= trans_volume * transaction.price
                self.holdings[transaction.symbol] += trans_volume
                self.tradable_holdings[transaction.symbol] += trans_volume
            else:
                self.tradable_cash += trans_volume * transaction.price
        for sell_id in transaction.sell_id:
            if sell_id != trans_order_id_to_notify:
                continue
            assert sell_id in self.lob_orders[transaction.symbol]
            limit_order = self.lob_orders[transaction.symbol][sell_id]
            trans_volume = transaction.volume if transaction.order_matched_volume is None else transaction.order_matched_volume[sell_id]
            limit_order.decrease_volume(trans_volume)
            assert limit_order.volume >= 0
            if limit_order.volume == 0:
                self.lob_orders[limit_order.symbol].pop(limit_order.order_id)
                self.lob_price_orders[limit_order.symbol][limit_order.price].pop(limit_order.order_id)
                del_order_ids.append((limit_order.symbol, limit_order.order_id))
            if transaction.type != "C":
                # sell order was matched: holding turn into cash
                self.cash += trans_volume * transaction.price
                self.tradable_cash += trans_volume * transaction.price
                self.holdings[transaction.symbol] -= trans_volume
            else:
                self.tradable_holdings[transaction.symbol] += trans_volume
        return del_order_ids

    def construct_valid_orders(
        self,
        time: Timestamp,
        symbol: str,
        type: str,
        price: int,
        volume: int,
        random: Optional[Random] = None,
    ) -> List[LimitOrder]:
        """Construct valid orders.

        Args
        ----
            time (Timestamp): the time of orders.
            symbol (str): the symbol of orders.
            type (str): the order type, valid types are "B", "S" or "C".
            price (int): the order price.
            volume (int): the order volume.

        Returns
        -------
            List[LimitOrder]: limit orders that try to satisfy the `symbol`, `type`, `price`, and `volume`.
            - For buy and sell order, it returns a single order with `type`, `price` and `volume`.
            - For cancel order, it first finds a valid price that closest to `price`,
            and then try to construct orders with total volume <= `volume`.

        """
        assert type in ["B", "S", "C"]
        if type in ["B", "S"]:
            return self._construct_buy_sell_order(time, symbol, type, price, volume)
        else:
            return self._construct_valid_cancel_orders(time, symbol, type, price, volume, random=random)

    def _construct_buy_sell_order(self, time: Timestamp, symbol: str, type: str, price: int, volume: int):
        """Construct buy or sell order based on type, price, volume."""
        order = LimitOrder(
            time=time,
            type=type,
            price=price,
            volume=volume,
            symbol=symbol,
            agent_id=self.agent_id,
            order_id=-1,
            cancel_type="None",
            cancel_id=-1,
            tag="",
        )
        return [order]

    def _get_closest_existing_price(self, pred_price: int, prices: List[int], random: Optional[Random] = None):
        min_dis = sys.maxsize
        if random is None:
            shuffle(prices)
        else:
            random.shuffle(prices)
        closest_price = -1
        for price in prices:
            dis = abs(price - pred_price)
            if dis < min_dis:
                min_dis = dis
                closest_price = price
        # if min_dis != 0:
        #     logging.warning(f"invalid price: {pred_price}, replaced with a closest price: {closest_price}")
        assert closest_price > 0
        return closest_price

    def _construct_valid_cancel_orders(self, time: Timestamp, symbol: str, type: str, price: int, volume: int, random: Optional[Random] = None):
        """Construct valid cancel orders.

        The order id to cancel is based on agent's existing orders on orderbook.
        """
        orders: List[LimitOrder] = []
        price_orders = self.lob_price_orders[symbol]
        valid_prices = [key for key, value in price_orders.items() if value]
        if not valid_prices:
            return orders

        price = self._get_closest_existing_price(pred_price=price, prices=valid_prices, random=random)
        state = self.symbol_states[symbol][State.__name__]
        lob = state.lob_snapshot
        assert lob is not None
        cancel_type: str = "None"
        if lob.bid_prices and price <= lob.bid_prices[0]:
            cancel_type = "B"
        elif lob.ask_prices and price >= lob.ask_prices[0]:
            cancel_type = "S"
        assert cancel_type != "None"

        existing_orders = list(price_orders[price].values())
        assert existing_orders
        left_volume = volume
        for order in existing_orders:
            assert left_volume >= 0
            if left_volume == 0:
                break
            min_vol = min(order.volume, left_volume)
            assert min_vol > 0
            left_volume -= min_vol

            order = LimitOrder(
                time=time,
                type=type,
                price=price,
                volume=min_vol,
                symbol=symbol,
                agent_id=self.agent_id,
                order_id=-1,
                cancel_type=cancel_type,
                cancel_id=order.order_id,
                tag="",
            )
            orders.append(order)
        assert orders
        return orders
