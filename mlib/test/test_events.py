# pyright: strict
from typing import List

from pandas import Timestamp

from mlib.core.engine import Engine
from mlib.core.event import (
    Event,
    MarketCloseEvent,
    MarketOpenEvent,
    create_exchange_events,
)
from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_Chinese_stock_exchange_config


def test_create_exchange_events():
    date = Timestamp("2020-01-01")
    symbols = ["000001"]
    config = create_Chinese_stock_exchange_config(date, symbols=symbols)
    events = create_exchange_events(config)
    engine = Engine(Exchange(config), verbose=True)
    for event in events:
        engine.push_event(event)
    print("start to handle events")
    pop_events: List[Event] = []
    while engine.has_event():
        event = engine._pop_event()  # type: ignore
        if pop_events:
            assert pop_events[-1].time <= event.time
        pop_events.append(event)
    assert isinstance(pop_events[0], MarketOpenEvent)
    assert isinstance(pop_events[-1], MarketCloseEvent)


if __name__ == "__main__":
    test_create_exchange_events()
