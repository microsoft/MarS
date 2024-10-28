from typing import List, Optional

from pandas import Timestamp

import mlib.core.event as event
from mlib.core.base_order import BaseOrder
from mlib.core.exchange_config import ExchangeConfig
from mlib.core.limit_order import LimitOrder
from mlib.core.transaction import Transaction


class Event:
    """The base class of events."""

    def __init__(self, time: Timestamp):
        self.time: Timestamp = time
        self.priority: int = 0
        self.event_id: int = -1

    def __lt__(self, other: "Event"):
        if self.time == other.time:
            if self.priority == other.priority:
                return self.event_id < other.event_id
            return self.priority < other.priority
        return self.time < other.time


class AgentStatesUpdateAndWakeup(Event):
    """Agent update states and wakeup."""

    def __init__(self, time: Timestamp, agent_id: int, wakeup_time: Timestamp, market_open_wakeup: bool):
        super().__init__(time)
        self.agent_id = agent_id
        self.wakeup_time = wakeup_time
        self.market_open_wakeup = market_open_wakeup
        self.priority = 100


class ExchangeReceiveOrdersEvent(Event):
    """Exchange receives order from agents."""

    def __init__(self, time: Timestamp, agent_id: int, orders: List[BaseOrder]):
        super().__init__(time)
        self.agent_id = agent_id
        self.orders: List[BaseOrder] = orders


class AgentReceiveTradingResultEvent(Event):
    """Agent receive trading result from exchange."""

    def __init__(
        self,
        time: Timestamp,
        agent_id: int,
        accepted_orders: List[LimitOrder],
        rejected_orders: List[BaseOrder],
        ignore_orders: List[BaseOrder],
        trans_info: Optional[Transaction],
        trans_order_id_to_notify: Optional[int],
    ):
        super().__init__(time)
        self.agent_id = agent_id
        self.accepted_orders = accepted_orders
        self.rejected_orders = rejected_orders
        self.ignore_orders = ignore_orders
        self.trans_info = trans_info
        self.trans_order_id_to_notify = trans_order_id_to_notify


class MarketOpenEvent(Event):
    """Market open."""


class MarketCloseEvent(Event):
    """Market close."""


class CallAuctionStartEvent(Event):
    """Call auction starts."""


class CallAuctionEndEvent(Event):
    """Call auction ends."""


class ContinuousAuctionStartEvent(Event):
    """Continuous auction starts."""


class ContinuousAuctionEndEvent(Event):
    """Continuous auction ends."""


def create_exchange_events(
    trade_config: ExchangeConfig,
    open_auction_event: bool = True,
    close_auction_event: bool = True,
):
    """Create exchange event based on trade config."""
    events: List[event.Event] = []
    # open event
    events.append(event.MarketOpenEvent(trade_config.open_time))

    # open auction
    if open_auction_event and trade_config.open_auction_start_time and trade_config.open_auction_end_time:
        events.append(event.CallAuctionStartEvent(trade_config.open_auction_start_time))
        events.append(event.CallAuctionEndEvent(trade_config.open_auction_end_time))

    # continuous auction
    if trade_config.continuous_auction_start_time and trade_config.continuous_auction_end_time:
        events.append(event.ContinuousAuctionStartEvent(trade_config.continuous_auction_start_time))
        events.append(event.ContinuousAuctionEndEvent(trade_config.continuous_auction_end_time))

    # close auction
    if close_auction_event and trade_config.close_auction_start_time and trade_config.close_auction_end_time:
        events.append(event.CallAuctionStartEvent(trade_config.close_auction_start_time))
        events.append(event.CallAuctionEndEvent(trade_config.close_auction_end_time))

    # close event
    events.append(event.MarketCloseEvent(trade_config.close_time))
    events = [event for event in events if event.time is not None]
    assert isinstance(events[0], event.MarketOpenEvent)
    assert isinstance(events[-1], event.MarketCloseEvent)
    return events
