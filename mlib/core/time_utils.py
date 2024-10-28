# pyright: strict
import datetime
from typing import Optional

from pandas import Timestamp


def is_in_period(start: Optional[Timestamp], end: Optional[Timestamp], time: Timestamp):
    if start is None or end is None:
        return False
    return start <= time <= end


def get_ts(date: datetime.date, hour: int, minute: int, second: int, microsecond: int = 0):
    return Timestamp(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour,
        minute=minute,
        second=second,
        microsecond=microsecond,
    )


def get_minute(ts: Timestamp):
    return Timestamp(year=ts.year, month=ts.month, day=ts.day, hour=ts.hour, minute=ts.minute)


def elapsed_minutes(start_time: Timestamp, end_time: Timestamp):
    """NOTE: this function is slow."""
    total_minutes: int = int((end_time - start_time).total_seconds() // 60)
    return total_minutes


DEFAULT_TIME = Timestamp("1900-01-01")
