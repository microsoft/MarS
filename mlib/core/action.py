from typing import List, NamedTuple, Optional

from pandas import Timestamp

from mlib.core.base_order import BaseOrder


class Action(NamedTuple):
    time: Timestamp
    agent_id: int
    orders: List[BaseOrder]
    next_wakeup_time: Optional[Timestamp]
