from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from pandas import Timedelta, Timestamp

from market_simulation.rollout.model_client import ModelClient
from market_simulation.states.order_state import Converter, OrderState, PredOrderInfo
from mlib.core.action import Action
from mlib.core.base_agent import BaseAgent
from mlib.core.observation import Observation


class BackgroundAgent(BaseAgent):
    """Order Agent."""

    def __init__(
        self,
        symbol: str,
        converter: Converter,
        start_time: Timestamp,
        end_time: Timestamp,
        model_client: ModelClient,
        init_agent: BaseAgent,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.converter = converter
        self.model_client = model_client
        self.num_pred_orders = 0
        self.planned_action: Optional[PredOrderInfo] = None
        self.init_agent = init_agent
        self.finish_init: bool = False

    def get_action(self, observation: Observation) -> Action:
        """Get action based on observation."""
        assert self.agent_id == observation.agent.agent_id
        time = observation.time
        if time < self.start_time:
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=self.start_time)

        if time > self.end_time:
            # end of simulation
            return Action(agent_id=self.agent_id, orders=[], time=time, next_wakeup_time=None)

        if not self.finish_init:
            self.finish_init = True
            self.init_base_info(self.init_agent)

        assert self.start_time <= time <= self.end_time

        time = observation.time

        if self.planned_action is not None:
            # use planned action to construct valid orders based on latest states.
            orders = self.construct_valid_orders(
                time=time,
                symbol=self.symbol,
                type=self.planned_action.order_type,
                price=self.planned_action.price,
                volume=self.planned_action.volume,
            )
            action = Action(
                agent_id=self.agent_id,
                orders=orders,  # type: ignore
                time=time,
                next_wakeup_time=time,  # wakeup immediately after order submission.
            )
            self.planned_action = None
            return action

        mid_price, state_vector, state = self.get_order_state()
        predictions = self.model_client.get_prediction(state_vector)
        assert predictions.size == 1
        order_index = predictions[0]
        pred_order = state.get_pred_order_info(order_index)
        wakeup_time = time + Timedelta(seconds=self.converter.order_interval.sample(pred_order.interval))
        self.planned_action = PredOrderInfo(
            order_type=pred_order.order_type,
            price=mid_price + int(round(self.converter.price_level.sample(pred_order.price))),
            volume=int(round(self.converter.pred_order_volume.sample(pred_order.volume))),
            interval=0,
        )
        action = Action(
            agent_id=self.agent_id,
            orders=[],  # delay order submission to next wakeup.
            time=time,
            next_wakeup_time=wakeup_time,  # wakeup time to submit orders.
        )
        self.num_pred_orders += 1
        return action

    def get_order_state(self) -> Tuple[int, npt.NDArray[np.int32], OrderState]:
        """Get order state from OrderState."""
        state = self.symbol_states[self.symbol][OrderState.__name__]
        assert isinstance(state, OrderState)
        assert state.latest_lob is not None
        mid_price = state.latest_lob.mid_price
        vector = state.to_vector()
        return mid_price, vector, state
