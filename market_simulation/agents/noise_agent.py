from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import pandas as pd

from market_simulation.states.trade_info_state import TradeInfoState
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent

if TYPE_CHECKING:
    from mlib.core.lob_snapshot import LobSnapshot
    from mlib.core.observation import Observation


class NoiseAgent(BaseAgent):
    """Noise agent, which generates random orders based on predefined distributions."""

    def __init__(
        self,
        symbol: str,
        init_price: int,
        interval_seconds: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        seed: int,
    ) -> None:
        super().__init__(
            init_cash=1000,
            communication_delay=0,
            computation_delay=0,
        )
        self.symbol = symbol
        self.init_price = init_price
        self.start_time = start_time
        self.end_time = end_time
        self.interval_seconds = interval_seconds
        # probabilities for order type, price and volume
        self.type_probs: dict[str, float] = {"B": 0.4, "S": 0.4, "C": 0.2}
        self.price_level_probs: dict[int, float] = {
            0: 0.2,
            100: 0.2,
            -100: 0.2,
            200: 0.1,
            -200: 0.1,
            300: 0.05,
            -300: 0.05,
            400: 0.025,
            -400: 0.025,
            500: 0.025,
            -500: 0.025,
        }
        self.volume_probs: dict[int, float] = {100: 0.5, 200: 0.3, 500: 0.15, 1000: 0.025, 2000: 0.025}
        self.rnd = random.Random(seed)

    def get_action(self, observation: Observation) -> Action:
        """Generate a random action based on the observation."""
        assert self.agent_id == observation.agent.agent_id, f"Agent ID mismatch: {self.agent_id} != {observation.agent.agent_id}"
        time = observation.time
        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)
        if time > self.end_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)
        lob = self.get_lob()
        base_price = lob.last_price if lob and lob.last_price > 0 else self.init_price
        # generate a random order by sampling from the predefined distributions
        orders = self.construct_valid_orders(
            time=time,
            symbol=self.symbol,
            type=self._sample(self.type_probs),
            price=base_price + self._sample(self.price_level_probs),
            volume=self._sample(self.volume_probs),
            random=self.rnd,
        )
        next_wakeup_time = time + pd.Timedelta(seconds=self.interval_seconds * self.rnd.uniform(0.5, 1.5))
        action = Action(
            agent_id=self.agent_id,
            orders=orders,  # type: ignore
            time=time,
            next_wakeup_time=next_wakeup_time,
        )
        return action

    def get_lob(self) -> LobSnapshot | None:
        """Get the latest LOB snapshot from the TradeInfoState."""
        state = self.symbol_states[self.symbol][TradeInfoState.__name__]
        assert isinstance(state, TradeInfoState)
        if not state.trade_infos:
            return None
        lob = state.trade_infos[-1].lob_snapshot
        return lob

    def _sample(self, probs: dict[Any, float]) -> Any:
        """Sample from a discrete distribution."""
        return self.rnd.choices(population=list(probs.keys()), weights=list(probs.values()), k=1)[0]
