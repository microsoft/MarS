# pyright: strict

from time import sleep
from typing import List

from pandas import Timestamp, date_range
from rich.progress import (
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class TimeProgress:
    """Show the time progress given start time and end time."""

    def __init__(
        self,
        start_time: Timestamp,
        end_time: Timestamp,
        description: str,
        unit: str = "min",
        refresh_per_second: float = 0.2,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.units: List[Timestamp] = list(date_range(start_time, end_time, freq=f"1{unit}"))
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            # BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            refresh_per_second=refresh_per_second,
        )
        self.task = self.progress.add_task(f"[green]{description}", total=len(self.units))
        self.description = description
        self.total_completed = 0

    def update(self, current_time: Timestamp):
        completed = 0
        # NOTE: here we avoid using (current_time - start_time).total_seconds() due to its poor performance.
        while self.units:
            if self.units[0] <= current_time:
                self.units.pop(0)
                completed += 1
            else:
                break
        if completed > 0:
            self.total_completed += completed
            self.progress.update(
                self.task,
                completed=self.total_completed,
                description=f"[green]{self.description}, current time: {current_time}",
            )


if __name__ == "__main__":
    start, end = Timestamp("2020-01-01"), Timestamp("2020-01-02")
    progress = TimeProgress(start, end, unit="min", description="test")
    seconds: List[Timestamp] = list(date_range(start, end, freq="1s"))
    with progress.progress:
        for second in seconds:
            progress.update(second)
            sleep(0.0001)
