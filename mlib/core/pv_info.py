# pyright: strict
from typing import List, NamedTuple


class PvInfo(NamedTuple):
    price: int
    volume: int

    @staticmethod
    def get_vwap(pv_infos: List["PvInfo"]):
        assert pv_infos
        total_money = sum([pv_info.price * pv_info.volume for pv_info in pv_infos])
        total_volume = sum([pv_info.volume for pv_info in pv_infos])
        assert total_money > 0
        assert total_volume > 0
        return total_money / total_volume
