# pyright: strict

from typing import Generator

from pandas import Timestamp

from mlib.core.action import Action
from mlib.core.engine import Engine
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
from mlib.core.observation import Observation
from mlib.utils.time_progress import TimeProgress


class Env(Engine):
    def __init__(self, exchange: Exchange, description: str = "", verbose: bool = False) -> None:
        super().__init__(exchange, description, verbose)

    def env(self) -> Generator[Observation, None, None]:
        self.is_generator = True
        start_time = Timestamp.now()
        time_progress: TimeProgress = TimeProgress(
            self.exchange.config.open_time, self.exchange.config.close_time, description=self.description, unit="s"
        )
        with time_progress.progress:
            while self.events:
                event = self._pop_event()
                yield from self._handle_event_generator(event=event)
                time_progress.update(event.time)
        self._log(f"finished processing {self._num_event} events in {(Timestamp.now() - start_time).total_seconds()}s.")

    def step(self, action: Action):
        assert self.is_generator
        self._on_receive_agent_action(action=action)

    def _handle_event_generator(self, event: Event):  # type: ignore
        assert event.event_id >= 0

        if isinstance(event, MarketOpenEvent):
            self._on_market_open_event(event)
        elif isinstance(event, MarketCloseEvent):
            self._on_market_close_event(event)
        elif isinstance(event, CallAuctionEndEvent):
            self._on_call_auction_end(event)
        elif isinstance(event, AgentStatesUpdateAndWakeup):
            yield from self._on_agent_states_update_and_wakeup_generator(event)
        elif isinstance(event, ExchangeReceiveOrdersEvent):
            self._on_exchange_receive_orders(event)
        elif isinstance(event, AgentReceiveTradingResultEvent):
            self._on_agent_receive_trading_result(event)
        else:
            if (
                isinstance(event, CallAuctionStartEvent)
                or isinstance(event, ContinuousAuctionStartEvent)
                or isinstance(event, ContinuousAuctionEndEvent)
            ):
                return
            raise ValueError(f"event without handler: {event}")

    def _on_agent_states_update_and_wakeup_generator(self, event: AgentStatesUpdateAndWakeup):
        observation = self._get_updated_agent_observation(event)
        yield observation

    def run(self) -> None:
        raise NotImplementedError("run() is only available for Engine")
