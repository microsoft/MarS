from typing import TYPE_CHECKING, NamedTuple

from pandas import Timestamp

if TYPE_CHECKING:
    from mlib.core.base_agent import BaseAgent


class Observation(NamedTuple):
    """Observation for the agent."""

    time: Timestamp
    agent: "BaseAgent"
    is_market_open_wakup: bool
