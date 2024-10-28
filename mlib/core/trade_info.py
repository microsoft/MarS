# pyright: strict
from typing import List

from mlib.core.limit_order import LimitOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.transaction import Transaction


class TradeInfo:
    def __init__(self, order: LimitOrder, trans: List[Transaction], lob_snapshot: LobSnapshot) -> None:
        self.order: LimitOrder = order
        self.transactions: List[Transaction] = trans
        self.lob_snapshot: LobSnapshot = lob_snapshot
