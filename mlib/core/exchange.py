# pyright: strict
from copy import deepcopy

from pandas import Timestamp

from mlib.core.base_order import BaseOrder
from mlib.core.exchange_config import ExchangeConfig
from mlib.core.limit_order import LimitOrder
from mlib.core.orderbook import Orderbook
from mlib.core.state import State
from mlib.core.time_utils import is_in_period
from mlib.core.trade_info import TradeInfo
from mlib.core.transaction import Transaction


class Exchange:
    """Exchange class."""

    def __init__(self, exchange_config: ExchangeConfig) -> None:
        self.config = exchange_config
        self._orderbooks: dict[str, Orderbook] = {}
        self.snapshot_levels = 10
        self._init_exchange()

    def market_open(self, time: Timestamp) -> None:
        """Market open."""
        # logging.info(f"market open at {time}")
        pass

    def market_close(self, time: Timestamp) -> None:
        """Market close."""
        # logging.info(f"market close at {time}")
        pass

    def _init_symbol_states(self) -> None:
        self._symbol_states: dict[str, dict[str, State]] = {}  # _symbol_states[symbol][state_cls_name]
        for symbol in self.config.symbols:
            self._symbol_states[symbol] = {}

    def register_state(self, state: State) -> None:
        """Register state."""
        name = state.__class__.__name__
        for symbol in self.config.symbols:
            self._symbol_states[symbol][name] = state

    def _init_exchange(self) -> None:
        self._orderbooks = {symbol: Orderbook(symbol) for symbol in self.config.symbols}
        self._init_symbol_states()
        self.register_state(State())
        self._cur_order_id: int = 1000000000

    def set_order_id(self, order: LimitOrder) -> None:
        """Set order ID."""
        order.set_order_id(self._cur_order_id)
        self._cur_order_id += 1

    def is_in_call_auction_period(self, time: Timestamp) -> bool:
        """Check if time is in call auction period."""
        if is_in_period(self.config.open_auction_start_time, self.config.open_auction_end_time, time):
            return True

        if is_in_period(self.config.close_auction_start_time, self.config.close_auction_end_time, time):
            return True
        return False

    def is_in_continuous_auction_period(self, time: Timestamp) -> bool:
        """Check if time is in continuous auction period."""
        return is_in_period(self.config.continuous_auction_start_time, self.config.continuous_auction_end_time, time)

    def submit_call_auction_order(self, base_order: BaseOrder) -> list[LimitOrder] | None:
        """Submit call auction order."""
        assert self.is_in_call_auction_period(base_order.time)
        orders: list[LimitOrder] = base_order.get_limit_orders(self._orderbooks[base_order.symbol])
        if not orders:
            return None
        for order in orders:
            if order.order_id < 0:
                self.set_order_id(order)
            self._orderbooks[order.symbol].add_call_auction_order(order.clone())
            # update state with call auction order
            trade_info = TradeInfo(
                order=order.clone(),
                trans=[],
                lob_snapshot=self._orderbooks[base_order.symbol].snapshot(self.snapshot_levels),
            )
            states = self._symbol_states[order.symbol]
            for v in states.values():
                v.on_call_auction_trading(trade_info)
        return orders

    def submit_continuous_auction_order(self, base_order: BaseOrder) -> list[TradeInfo] | None:
        """Submit continuous auction order."""
        trade_infos: list[TradeInfo] = []
        orders: list[LimitOrder] = base_order.get_limit_orders(self._orderbooks[base_order.symbol])
        if not orders:
            return None

        for order in orders:
            if order.order_id < 0:
                self.set_order_id(order)

            trade_info = self._orderbooks[order.symbol].update(order)
            states = self._symbol_states[order.symbol]
            for v in states.values():
                v.on_trading(trade_info)
            trade_infos.append(trade_info)
        return trade_infos

    def match_call_auction_orders(self, time: Timestamp) -> dict[str, list[Transaction]]:
        """Match call auction orders."""
        results: dict[str, list[Transaction]] = {}
        for symbol in self.config.symbols:
            results[symbol] = self._match_call_auction_orders(time, symbol)
        return results

    def _match_call_auction_orders(self, time: Timestamp, symbol: str) -> list[Transaction]:
        states = self._symbol_states[symbol]
        if is_in_period(self.config.open_auction_start_time, self.config.open_auction_end_time, time):
            cancel_transactions, match_transaction = self._orderbooks[symbol].match_call_auction_orders(time, "OPEN")
            for state in states.values():
                state.on_open(
                    cancel_transactions=cancel_transactions,
                    match_trans=match_transaction,
                    lob_snapshot=self._orderbooks[symbol].snapshot(self.snapshot_levels),
                )
        else:
            assert is_in_period(self.config.close_auction_start_time, self.config.close_auction_end_time, time)
            cancel_transactions, match_transaction = self._orderbooks[symbol].match_call_auction_orders(time, "CLOSE")
            assert not cancel_transactions
            for state in states.values():
                state.on_close(
                    match_trans=match_transaction,
                    close_orderbook=self._orderbooks[symbol],
                    lob_snapshot=self._orderbooks[symbol].snapshot(self.snapshot_levels),
                )
        transactions: list[Transaction] = []
        transactions.extend(cancel_transactions)
        if match_transaction is not None:
            transactions.append(match_transaction)
        return transactions

    def states_snapshot(self) -> dict[str, dict[str, State]]:
        """Create a copy of current states.

        For a better performance, we should not call this function frequently.
        """
        return deepcopy(self._symbol_states)

    def states(self) -> dict[str, dict[str, State]]:
        """Get states."""
        return self._symbol_states

    def get_lob(self, symbol: str) -> Orderbook:
        """Get lobs."""
        return self._orderbooks[symbol]
