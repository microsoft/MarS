from mlib.core.action import Action
import pandas as pd
from mlib.core.base_agent import BaseAgent
import numpy as np
from typing import Dict, Any
from market_simulation.states.trade_info_state import TradeInfoState
from mlib.core.observation import Observation


class NoiseAgent(BaseAgent):
    def __init__(
        self,
        symbol: str,
        init_price: int,
        interval_seconds: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ):
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
        self.type_probs: Dict[str, float] = {"B": 0.5, "S": 0.5}
        self.price_level_probs: Dict[int, float] = {0: 0.35, 100: 0.35, -100: 0.15, 200: 0.15}
        self.volume_probs: Dict[int, float] = {100: 0.5, 200: 0.3, 500: 0.15, 1000: 0.025, 2000: 0.025}

    def get_action(self, observation: Observation) -> Action:
        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)
        if time > self.end_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)
        lob = self.get_lob()
        mid_price = lob.bid_prices[0] if lob and lob.bid_prices else self.init_price
        # generate a random order by sampling from the predefined distributions
        orders = self.construct_valid_orders(
            time=time,
            symbol=self.symbol,
            type=self._sample(self.type_probs),
            price=mid_price + self._sample(self.price_level_probs),
            volume=self._sample(self.volume_probs),
        )
        next_wakeup_time = time + pd.Timedelta(seconds=self.interval_seconds)
        action = Action(
            agent_id=self.agent_id,
            orders=orders,  # type: ignore
            time=time,
            next_wakeup_time=next_wakeup_time,
        )
        return action

    def get_lob(self):
        state = self.symbol_states[self.symbol][TradeInfoState.__name__]
        assert isinstance(state, TradeInfoState)
        if not state.trade_infos:
            return None
        lob = state.trade_infos[-1].lob_snapshot
        return lob

    def _sample(self, probs: Dict[Any, float]) -> Any:
        """Sample from a discrete distribution."""
        return np.random.choice(list(probs.keys()), p=list(probs.values()))
