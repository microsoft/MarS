# pyright: strict
import heapq
import logging
from typing import TYPE_CHECKING

from pandas import Timedelta, Timestamp

from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent
from mlib.core.event import (
    AgentReceiveTradingResultEvent,
    AgentStatesUpdateAndWakeup,
    CallAuctionEndEvent,
    CallAuctionStartEvent,
    ContinuousAuctionEndEvent,
    ContinuousAuctionStartEvent,
    Event,
    ExchangeReceiveOrdersEvent,
    MarketCloseEvent,
    MarketOpenEvent,
)
from mlib.core.exchange import Exchange
from mlib.core.limit_order import LimitOrder
from mlib.core.observation import Observation
from mlib.core.transaction import Transaction
from mlib.utils.time_progress import TimeProgress

if TYPE_CHECKING:
    from mlib.core.base_order import BaseOrder


class Engine:
    """Engine to support async events."""

    def __init__(self, exchange: Exchange, description: str = "", *, verbose: bool = False) -> None:
        self.events: list[Event] = []
        self._num_event = 0
        self.verbose = verbose
        self.exchange: Exchange = exchange
        self.description = description
        self.agents: dict[int, BaseAgent] = {}
        self.order_owner: dict[tuple[str, int], int] = {}  # symbol, order_id -> agent_id

    def has_event(self) -> bool:
        """Has any event."""
        return len(self.events) > 0

    def register_agent(self, agent: BaseAgent) -> None:
        """Register agent."""
        num_agent = len(self.agents)
        agent.agent_id = num_agent
        self.agents[agent.agent_id] = agent

    def push_event(self, event: Event) -> None:
        """Push event."""
        max_event_id = 1000000000
        if isinstance(event, MarketCloseEvent):
            event.event_id = max_event_id
        else:
            event.event_id = self._num_event
            self._num_event += 1
            assert event.event_id < max_event_id
        heapq.heappush(self.events, event)
        self._log(f"received new event, type: {event.__class__.__name__}, {event.time}")

    def push_events(self, events: list[Event]) -> None:
        """Push events."""
        for event in events:
            self.push_event(event)

    def run(self) -> None:
        """Run event engine."""
        start_time = Timestamp.now()
        time_progress: TimeProgress = TimeProgress(
            self.exchange.config.open_time, self.exchange.config.close_time, description=self.description, unit="s"
        )
        with time_progress.progress:
            while self.events:
                event = self._pop_event()
                self._handle_event(event)
                time_progress.update(event.time)
        self._log(f"finished processing {self._num_event} events in {(Timestamp.now() - start_time).total_seconds()}s.")

    def _pop_event(self) -> Event:
        """Pop event."""
        assert self.events
        event: Event = heapq.heappop(self.events)
        self._log(f"handling event, type: {event.__class__.__name__}, {event.time}")
        return event

    def _handle_event(self, event: Event) -> None:
        assert event.event_id >= 0

        if isinstance(event, MarketOpenEvent):
            self._on_market_open_event(event)
        elif isinstance(event, MarketCloseEvent):
            self._on_market_close_event(event)
        elif isinstance(event, CallAuctionEndEvent):
            self._on_call_auction_end(event)
        elif isinstance(event, AgentStatesUpdateAndWakeup):
            self._on_agent_states_update_and_wakeup(event)
        elif isinstance(event, ExchangeReceiveOrdersEvent):
            self._on_exchange_receive_orders(event)
        elif isinstance(event, AgentReceiveTradingResultEvent):
            self._on_agent_receive_trading_result(event)
        else:
            if isinstance(event, (CallAuctionStartEvent, ContinuousAuctionStartEvent, ContinuousAuctionEndEvent)):
                return
            raise ValueError(f"event without handler: {event}")

    def _add_order_owner_info(self, symbol: str, order_id: int, agent_id: int) -> None:
        key = (symbol, order_id)
        assert key not in self.order_owner, f"key existed: {key}"
        self.order_owner[key] = agent_id

    def _del_order_owner_info(self, symbol: str, order_id: int) -> None:
        key = (symbol, order_id)
        assert key in self.order_owner
        self.order_owner.pop(key)

    def _on_agent_receive_trading_result(self, event: AgentReceiveTradingResultEvent) -> None:
        agent = self.agents[event.agent_id]
        if event.accepted_orders:
            agent.on_order_accepted(event.time, event.accepted_orders)
        if event.rejected_orders:
            agent.on_order_rejected(event.time, event.rejected_orders)
        if event.ignore_orders:
            agent.on_order_ignored(event.time, event.ignore_orders)
        if event.trans_info:
            assert event.trans_order_id_to_notify is not None
            del_order_ids = agent.on_order_executed(event.time, event.trans_info, event.trans_order_id_to_notify)
            for symbol, order_id in del_order_ids:
                self._del_order_owner_info(symbol, order_id)

    def _on_receive_agent_action(self, action: Action) -> None:
        agent = self.agents[action.agent_id]
        wakeup_time = action.time
        if action.orders:
            orders: list[BaseOrder] = action.orders
            for order in orders:
                order.set_agent_id(agent.agent_id)
            self.push_event(
                ExchangeReceiveOrdersEvent(
                    time=wakeup_time + agent.computation_delay + agent.communication_delay,
                    agent_id=action.agent_id,
                    orders=orders,
                )
            )

        # add next wakeup event
        if action.next_wakeup_time is not None:
            next_wakeup_time = action.next_wakeup_time
            self._check_states_update_time(wakeup_time, next_wakeup_time, agent.communication_delay)
            next_states_update_time = next_wakeup_time - agent.communication_delay
            self.push_event(
                AgentStatesUpdateAndWakeup(
                    time=next_states_update_time,
                    agent_id=agent.agent_id,
                    wakeup_time=next_wakeup_time,
                    market_open_wakeup=False,
                )
            )

    def _get_updated_agent_observation(self, event: AgentStatesUpdateAndWakeup) -> Observation:
        agent: BaseAgent = self.agents[event.agent_id]

        # update status
        states_update_time = event.time
        assert agent.states_update_time <= states_update_time
        agent.on_states_update(states_update_time, self.exchange.states())
        if not event.market_open_wakeup:
            assert agent.states_update_time + agent.communication_delay == event.wakeup_time

        # contruct observation
        wakeup_time = event.wakeup_time
        observation = Observation(
            time=wakeup_time,
            agent=agent,
            is_market_open_wakup=event.market_open_wakeup,
        )
        return observation

    def _on_agent_states_update_and_wakeup(self, event: AgentStatesUpdateAndWakeup) -> None:
        """Update states and immediately get orders based on latest states and wakeup time.

        1. update states.
        2. wakeup and get orders based on latest states and wakeup time.
        3. set next wakeup event.

        """
        observation = self._get_updated_agent_observation(event=event)
        # wakeup and get orders
        agent = self.agents[event.agent_id]
        action = agent.get_action(observation=observation)
        if action.orders:
            assert not event.market_open_wakeup, "should not generate order for the market-open wakeup"

        self._on_receive_agent_action(action)

    def _on_exchange_receive_call_auction_orders(self, event: ExchangeReceiveOrdersEvent) -> None:
        assert self.exchange.is_in_call_auction_period(event.time)
        agent = self.agents[event.agent_id]
        accepted_orders: list[LimitOrder] = []
        rejected_orders: list[BaseOrder] = []
        for base_order in event.orders:
            limit_orders = self.exchange.submit_call_auction_order(base_order)
            if limit_orders is None:
                rejected_orders.append(base_order)
            else:
                accepted_orders.extend(limit_orders)
        self.push_event(
            AgentReceiveTradingResultEvent(
                time=event.time + agent.communication_delay,
                agent_id=event.agent_id,
                accepted_orders=accepted_orders,
                rejected_orders=rejected_orders,
                ignore_orders=[],
                trans_info=None,
                trans_order_id_to_notify=None,
            )
        )
        self._update_order_owner(accepted_orders)

    def _update_order_owner(self, accepted_orders: list[LimitOrder]) -> None:
        for order in accepted_orders:
            self._add_order_owner_info(order.symbol, order.order_id, order.agent_id)

    def _on_exchange_receive_continuous_auction_orders(self, event: ExchangeReceiveOrdersEvent) -> None:
        agent = self.agents[event.agent_id]
        accepted_orders: list[LimitOrder] = []
        rejected_orders: list[BaseOrder] = []
        trans_infos: list[Transaction] = []
        assert self.exchange.is_in_continuous_auction_period(event.time)
        for base_order in event.orders:
            order_trade_infos = self.exchange.submit_continuous_auction_order(base_order)
            if order_trade_infos is None:
                rejected_orders.append(base_order)
            else:
                for trade_info in order_trade_infos:
                    accepted_orders.append(trade_info.order.clone())
                    trans_infos.extend(trade_info.transactions)
        self.push_event(
            AgentReceiveTradingResultEvent(
                time=event.time + agent.communication_delay,
                agent_id=event.agent_id,
                accepted_orders=accepted_orders,
                rejected_orders=rejected_orders,
                ignore_orders=[],
                trans_info=None,
                trans_order_id_to_notify=None,
            )
        )
        self._update_order_owner(accepted_orders)
        for transaction in trans_infos:
            self.notify_agents_on_order_executed(event.time, transaction)

    def _on_exchange_receive_orders_out_of_trading_period(self, event: ExchangeReceiveOrdersEvent) -> None:
        agent = self.agents[event.agent_id]
        self.push_event(
            AgentReceiveTradingResultEvent(
                time=event.time + agent.communication_delay,
                agent_id=event.agent_id,
                accepted_orders=[],
                rejected_orders=[],
                ignore_orders=event.orders,
                trans_info=None,
                trans_order_id_to_notify=None,
            )
        )

    def _on_exchange_receive_orders(self, event: ExchangeReceiveOrdersEvent) -> None:
        if self.exchange.is_in_call_auction_period(event.time):
            self._on_exchange_receive_call_auction_orders(event)
        elif self.exchange.is_in_continuous_auction_period(event.time):
            self._on_exchange_receive_continuous_auction_orders(event)
        else:
            self._on_exchange_receive_orders_out_of_trading_period(event)

    def _on_market_close_event(self, event: MarketCloseEvent) -> None:
        self.exchange.market_close(event.time)
        for agent in self.agents.values():
            agent.on_states_update(event.time, self.exchange.states())
            agent.on_market_close(event.time)

    def _on_call_auction_end(self, event: CallAuctionEndEvent) -> None:
        symbol_transactions = self.exchange.match_call_auction_orders(event.time)
        for transactions in symbol_transactions.values():
            for transaction in transactions:
                self.notify_agents_on_order_executed(event.time, transaction)

    def notify_agents_on_order_executed(self, time: Timestamp, transaction: Transaction) -> None:
        """Notify agents on order executed."""
        symbol = transaction.symbol
        order_ids: list[int] = []
        order_ids.extend(transaction.buy_id)
        order_ids.extend(transaction.sell_id)
        for order_id in order_ids:
            key = (symbol, order_id)
            assert key in self.order_owner
            agent_id = self.order_owner[key]
            agent = self.agents[agent_id]
            self.push_event(
                AgentReceiveTradingResultEvent(
                    time=time + agent.communication_delay,
                    agent_id=agent_id,
                    accepted_orders=[],
                    rejected_orders=[],
                    ignore_orders=[],
                    trans_info=transaction,
                    trans_order_id_to_notify=order_id,
                )
            )

    def _check_states_update_time(
        self,
        current_time: Timestamp,
        next_wakeup_time: Timestamp,
        communication_delay: Timedelta,
    ) -> None:
        states_update_time = next_wakeup_time - communication_delay
        if states_update_time < current_time:
            raise ValueError(
                f"The earlist wakeup time should be: {current_time + communication_delay}."
                f"Reason: due to agent's network latency: {communication_delay}, "
                f"if we want to wakeup at {next_wakeup_time}, "
                f"we should update states at: {next_wakeup_time - communication_delay}, "
                f"which is earlier than current time: {current_time}"
            )

    def _on_market_open_event(self, event: MarketOpenEvent) -> None:
        self.exchange.market_open(event.time)
        for agent in self.agents.values():
            agent.on_market_open(event.time, self.exchange.config.symbols)
            wakeup_time: Timestamp = event.time
            self.push_event(
                AgentStatesUpdateAndWakeup(
                    time=wakeup_time,
                    agent_id=agent.agent_id,
                    wakeup_time=wakeup_time,
                    market_open_wakeup=True,
                )
            )

    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        logging.info(msg)
