from typing import NamedTuple

from pandas import Timestamp

from mlib.core.base_order import BaseOrder


class Action(NamedTuple):
    """Action to be executed by the agent."""

    time: Timestamp
    agent_id: int
    orders: list[BaseOrder]
    next_wakeup_time: Timestamp | None
