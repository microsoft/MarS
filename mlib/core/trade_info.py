# pyright: strict

from mlib.core.limit_order import LimitOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.transaction import Transaction


class TradeInfo:
    """Trade information, including order, transactions, and LOB snapshot."""

    def __init__(self, order: LimitOrder, trans: list[Transaction], lob_snapshot: LobSnapshot) -> None:
        self.order: LimitOrder = order
        self.transactions: list[Transaction] = trans
        self.lob_snapshot: LobSnapshot = lob_snapshot
