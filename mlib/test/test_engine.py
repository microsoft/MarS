# pyright: strict
from pandas import Timestamp

from mlib.core.engine import Engine
from mlib.core.event import Event, create_exchange_events
from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.time_utils import get_ts


def test_priority_queue() -> None:
    """Test priority queue."""
    engine: Engine = Engine(exchange=None, verbose=True)  # type: ignore
    now = Timestamp("2020-02-02")
    yesterday = Timestamp("2020-02-01")
    engine.push_event(Event(now))
    engine.push_event(Event(yesterday))
    pop_event0 = engine._pop_event()  # type: ignore  # noqa: SLF001
    pop_event1 = engine._pop_event()  # type: ignore  # noqa: SLF001
    assert pop_event0.time == yesterday
    assert pop_event0.event_id == 1
    assert pop_event1.time == now
    assert pop_event1.event_id == 0


def test_engine_run() -> None:
    """Run engine test."""
    date = Timestamp("2020-01-01")
    symbols = ["000001"]
    config = create_exchange_config_without_call_auction(
        market_open=get_ts(date, 9, 30, 0),
        market_close=get_ts(date, 15, 0, 0),
        symbols=symbols,
    )
    events = create_exchange_events(config)
    engine = Engine(Exchange(config), verbose=True)
    engine.push_events(events=events)
    engine.run()


if __name__ == "__main__":
    test_priority_queue()
    test_engine_run()
