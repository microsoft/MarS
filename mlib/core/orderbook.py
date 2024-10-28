# pyright: strict
from typing import Dict, List, Set

import pandas as pd
from pandas import Timestamp

from mlib.core.level import Level
from mlib.core.limit_order import LimitOrder
from mlib.core.lob_snapshot import LobSnapshot
from mlib.core.trade_info import TradeInfo
from mlib.core.transaction import Transaction


class Orderbook:
    """Orderbook class."""

    def __init__(self, symbol: str) -> None:
        self.bids: List[Level] = []
        self.asks: List[Level] = []
        self.call_auction_orders: List[LimitOrder] = []
        self.time: pd.Timestamp
        self.last_price = -1
        self.symbol = symbol

    def add_call_auction_order(self, order: LimitOrder):
        self.time = order.time
        self.call_auction_orders.append(order)

    def match_call_auction_orders(self, time: Timestamp, type: str):
        assert type in ["OPEN", "CLOSE"]
        self.time = time
        cancel_transactions = self._del_canceled_call_auction_orders()
        match_transaction = self._macth_call_auction_orders(time, type)
        if match_transaction:
            self.update_last_price(match_transaction)
        return cancel_transactions, match_transaction

    def update(self, order: LimitOrder):
        """Update orderbook with continuous auction order."""
        assert not self.call_auction_orders
        self.time = order.time
        trans = self._update_with_normal_order(order.clone())
        for transaction in trans:
            self.update_last_price(transaction)
        trade_info = TradeInfo(order, trans, self.snapshot())
        return trade_info

    def update_last_price(self, transaction: Transaction):
        if transaction.type == "B" or transaction.type == "S":
            self.last_price = transaction.price

    def snapshot(self, level: int = 10):
        asks = self.asks[:level]
        bids = self.bids[:level]
        snapshot = LobSnapshot(
            time=self.time,
            max_level=level,
            last_price=self.last_price,
            ask_prices=[x.price for x in asks],
            ask_volumes=[x.volume for x in asks],
            bid_prices=[x.price for x in bids],
            bid_volumes=[x.volume for x in bids],
        )
        return snapshot

    def get_best_k_ask_bid(self, k: int = 5):
        ask_k = -1
        bid_k = -1
        if len(self.asks) >= k:
            ask_prices: List[int] = [x.price for x in self.asks]
            ask_k = ask_prices[k - 1]
        if len(self.bids) >= k:
            bid_prices: List[int] = [x.price for x in self.bids]
            bid_k = bid_prices[k - 1]
        return ask_k, bid_k

    def get_price_of_order_id(self, order_id: int):
        levels: List[Level] = []
        levels.extend(self.asks)
        levels.extend(self.bids)

        for level in levels:
            if level.has_order_id(order_id):
                return level.price

        for order in self.call_auction_orders:
            if order.order_id == order_id:
                return order.price

        raise RuntimeError(f"order id {order_id} not existed in orderbook/call_auction_orders.")

    def _clear_call_auction_order(self):
        for order in self.call_auction_orders:
            trans = self._update_with_normal_order(order)
            assert not trans
        self.call_auction_orders.clear()

    def _macth_call_auction_orders(self, time: Timestamp, type: str):
        auction_orders = self.call_auction_orders
        # extend
        for level in self.asks:
            auction_orders.extend(level.orders)
        for level in self.bids:
            auction_orders.extend(level.orders)
        # sort
        auction_orders.sort(key=lambda x: x.order_id)
        sell_levels: Dict[int, Level] = {}
        buy_levels: Dict[int, Level] = {}
        self.asks = []
        self.bids = []

        if not auction_orders:
            return None
        # group_orders
        sell_levels, buy_levels, prices = _update_levels(auction_orders, sell_levels=sell_levels, buy_levels=buy_levels)

        # get call auction price
        (
            _,
            max_vol_price,
            max_vol_equal_price_sell_vol,
            max_vol_equal_price_buy_vol,
        ) = _get_call_auction_final_price(sell_levels, buy_levels, prices)

        if max_vol_price <= 0:
            self._clear_call_auction_order()
            return None

        # clear auction and update orderbook
        transaction = self._match_auction_orders_and_update_orderbook(
            auction_orders, max_vol_price, max_vol_equal_price_sell_vol, max_vol_equal_price_buy_vol, time, type
        )
        self.call_auction_orders.clear()
        return transaction

    def _match_auction_orders_and_update_orderbook(
        self,
        auction_orders: List[LimitOrder],
        max_vol_price: int,
        max_vol_equal_price_sell_vol: int,
        max_vol_equal_price_buy_vol: int,
        time: Timestamp,
        type: str,
    ):
        bid_ids: List[int] = []
        ask_ids: List[int] = []
        total_volume = 0
        order_matched_volume: Dict[int, int] = {}
        for order in auction_orders:
            if order.is_buy and order.price > max_vol_price:
                bid_ids.append(order.order_id)
                total_volume += order.volume
                order_matched_volume[order.order_id] = order.volume
            elif order.is_sell and order.price < max_vol_price:
                ask_ids.append(order.order_id)
                order_matched_volume[order.order_id] = order.volume
            elif order.is_buy and order.price == max_vol_price:
                min_vol = min(order.volume, max_vol_equal_price_buy_vol)
                if min_vol > 0:
                    max_vol_equal_price_buy_vol -= min_vol
                    order.decrease_volume(min_vol)
                    order_matched_volume[order.order_id] = min_vol
                    bid_ids.append(order.order_id)
                    total_volume += min_vol
                if order.volume > 0:
                    trans = self._update_with_normal_order(order.clone())
                    assert not trans
            elif order.is_sell and order.price == max_vol_price:
                min_vol = min(order.volume, max_vol_equal_price_sell_vol)
                if min_vol > 0:
                    max_vol_equal_price_sell_vol -= min_vol
                    order.decrease_volume(min_vol)
                    order_matched_volume[order.order_id] = min_vol
                    ask_ids.append(order.order_id)
                if order.volume > 0:
                    trans = self._update_with_normal_order(order.clone())
                    assert not trans
            else:
                trans = self._update_with_normal_order(order)
                assert not trans

        transaction = Transaction(
            symbol=self.symbol,
            time=time,
            type=type,
            price=max_vol_price,
            volume=total_volume,
            buy_id=bid_ids,
            sell_id=ask_ids,
            order_matched_volume=order_matched_volume,
        )
        return transaction

    def _find_matched_index(self, price: int, levels: List[Level]):
        index = -1
        for i, level in enumerate(levels):
            if level.price == price:
                index = i
                break
        return index

    def _clear_levels(self, level_indexes: List[int], levels: List[Level]):
        for index in reversed(level_indexes):
            levels.pop(index)

    def _add_to_level(self, order: LimitOrder, levels: List[Level], ascending: bool):
        assert order.volume > 0
        for i, level in enumerate(levels):
            if order.price == level.price:
                level.add_new_order(order)
                return
            if ascending and level.price > order.price or not ascending and level.price < order.price:
                levels.insert(i, Level(order.price, order.volume, [order]))
                return
        levels.append(Level(order.price, order.volume, [order]))

    def _update_with_normal_order(self, order: LimitOrder):
        assert order.volume > 0
        assert order.price >= 0
        transactions: List[Transaction] = []
        if order.is_cancel_buy:
            index = self._find_matched_index(order.price, self.bids)
            assert index >= 0
            assert order.price != 0
            trans = self.bids[index].update_with_cancel_order(order)
            transactions.append(trans)
            if self.bids[index].volume == 0:
                self.bids.pop(index)
            return transactions
        elif order.is_cancel_sell:
            index = self._find_matched_index(order.price, self.asks)
            assert index >= 0
            assert order.price != 0
            trans = self.asks[index].update_with_cancel_order(order)
            transactions.append(trans)
            if self.asks[index].volume == 0:
                self.asks.pop(index)
            return transactions
        elif order.is_buy:
            levels = self.asks
            empty_levels: List[int] = []
            for index_level, level in enumerate(levels):
                if level.price <= order.price:
                    if level.volume > 0:
                        (
                            order,
                            matched_volume,
                            matched_details,
                        ) = level.update_with_clear_order(order)
                        assert matched_volume > 0
                        for id, vol in matched_details:
                            transactions.append(
                                Transaction(
                                    symbol=self.symbol,
                                    time=order.time,
                                    type="B",
                                    price=level.price,
                                    volume=vol,
                                    buy_id=[order.order_id],
                                    sell_id=[id],
                                )
                            )
                        if level.volume == 0:
                            empty_levels.append(index_level)
                        if order.volume == 0:
                            break
                else:
                    break
            self._clear_levels(empty_levels, levels)
            if order.volume > 0:
                self._add_to_level(order, self.bids, ascending=False)

        else:
            assert order.is_sell
            levels = self.bids
            empty_levels = []
            for index_level, level in enumerate(levels):
                if level.price >= order.price:
                    if level.volume > 0:
                        (
                            order,
                            matched_volume,
                            matched_details,
                        ) = level.update_with_clear_order(order)
                        assert matched_volume > 0
                        for id, vol in matched_details:
                            transactions.append(
                                Transaction(
                                    symbol=self.symbol,
                                    time=order.time,
                                    type="S",
                                    price=level.price,
                                    volume=vol,
                                    buy_id=[id],
                                    sell_id=[order.order_id],
                                )
                            )
                        if level.volume == 0:
                            empty_levels.append(index_level)
                        if order.volume == 0:
                            break
                else:
                    break
            self._clear_levels(empty_levels, levels)
            if order.volume > 0:
                self._add_to_level(order, self.asks, ascending=True)
        return transactions

    def _del_canceled_call_auction_orders(self):
        num_cancel = 0
        orders = self.call_auction_orders
        transactions: List[Transaction] = []
        for i in range(len(orders)):
            order = orders[i]
            if not order.is_cancel:
                continue
            num_cancel += 1
            pre_indexes = [i for i in range(0, i) if orders[i].order_id == order.cancel_id]
            assert len(pre_indexes) == 1
            pre_index = pre_indexes[0]
            pre_order = orders[pre_index]
            assert pre_order.price == order.price
            assert pre_order.volume >= order.volume
            pre_order.decrease_volume(order.volume)
            transactions.append(
                Transaction(
                    symbol=self.symbol,
                    time=order.time,
                    type="C",
                    price=0,
                    volume=order.volume,
                    buy_id=[pre_order.order_id] if pre_order.is_buy else [],
                    sell_id=[pre_order.order_id] if pre_order.is_sell else [],
                )
            )
            order.decrease_volume(order.volume)
        self.call_auction_orders = [x for x in orders if x.volume > 0]
        return transactions


def _update_levels(auction_orders: List[LimitOrder], sell_levels: Dict[int, Level], buy_levels: Dict[int, Level]):
    prices: Set[int] = set()
    for level in sell_levels.values():
        prices.add(level.price)

    for level in buy_levels.values():
        prices.add(level.price)

    for order in auction_orders:
        assert not order.is_cancel
        prices.add(order.price)
        if order.is_buy:
            if order.price not in buy_levels:
                buy_levels[order.price] = Level(order.price, 0, [])
            buy_levels[order.price].add_new_order(order)
        else:
            if order.price not in sell_levels:
                sell_levels[order.price] = Level(order.price, 0, [])
            sell_levels[order.price].add_new_order(order)

    return sell_levels, buy_levels, prices


def _get_call_auction_final_price(
    sell_levels: Dict[int, Level],
    buy_levels: Dict[int, Level],
    prices: Set[int],
):
    max_vol = 0
    max_vol_price = -1
    max_vol_equal_price_sell_vol = -1
    max_vol_equal_price_buy_vol = -1
    for price in prices:
        equal_price_sell_vol = 0
        equal_price_buy_vol = 0
        buy_vol = 0
        sell_vol = 0
        for buy_level in buy_levels.values():
            if buy_level.price > price:
                buy_vol += buy_level.volume
            elif buy_level.price == price:
                equal_price_buy_vol = buy_level.volume
        for sell_level in sell_levels.values():
            if sell_level.price < price:
                sell_vol += sell_level.volume
            elif sell_level.price == price:
                equal_price_sell_vol = sell_level.volume
        if buy_vol - sell_vol < 0 and buy_vol + equal_price_buy_vol - sell_vol < 0:
            continue
        if buy_vol - sell_vol > 0 and buy_vol - sell_vol - equal_price_sell_vol > 0:
            continue
        total_vol = min(buy_vol + equal_price_buy_vol, sell_vol + equal_price_sell_vol)
        if total_vol > max_vol:
            max_vol = total_vol
            max_vol_price = price
            max_vol_equal_price_sell_vol = total_vol - sell_vol
            max_vol_equal_price_buy_vol = total_vol - buy_vol
            assert 0 <= max_vol_equal_price_sell_vol <= equal_price_sell_vol
            assert 0 <= max_vol_equal_price_buy_vol <= equal_price_buy_vol
            assert (
                max_vol_equal_price_buy_vol - equal_price_buy_vol == 0
                or max_vol_equal_price_sell_vol - equal_price_sell_vol == 0
            )
            # logging.info(f'found new max vol: {max_vol}, price: {max_vol_price}, equal_sell_vol: {max_vol_equal_price_sell_vol}, equal_buy_vol: {max_vol_equal_price_buy_vol}')
    return (
        max_vol,
        max_vol_price,
        max_vol_equal_price_sell_vol,
        max_vol_equal_price_buy_vol,
    )
