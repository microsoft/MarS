import random
from typing import Counter  # noqa: UP035

import numpy as np


class BinConverter:
    """Create bins, get bin index and sample from bin index."""

    @staticmethod
    def create_from_values(values: list[float], num_bins: int, num_sample_per_bin: int = 100) -> "BinConverter":
        """Create bin converter from values."""
        converter = BinConverter()
        converter.init(values=values, num_bins=num_bins, num_sample_per_bin=num_sample_per_bin)
        return converter

    def init(self, values: list[float], num_bins: int, num_sample_per_bin: int = 100) -> None:
        """Initialize bin converter."""
        random.shuffle(values)
        values = values[:1000000]
        self._create_bins(values, num_bins)
        self._create_sample_probs(values, num_sample_per_bin)

    def _create_bins(self, values: list[float], num_bins: int) -> None:
        assert num_bins > 1
        values.sort()
        self.num_bins: int = num_bins
        value_freq: Counter[float] = Counter(values)
        avg_bin_sample_count = len(values) / num_bins

        # extract single items that can occupy a single bin
        single_item_bins: set[float] = set()
        for value, count in value_freq.items():
            if count > avg_bin_sample_count:
                single_item_bins.add(value)

        min_value = min(values)
        if min_value in single_item_bins:
            min_value = min_value - 1

        # remove items that are already in single-item bins
        values = [x for x in values if x not in single_item_bins]
        assert values

        num_values = len(values)
        available_bins = num_bins - len(single_item_bins) + 1
        cur_index = 0
        bins: list[float] = [min_value, *list(single_item_bins)]
        while available_bins > 0 and cur_index < num_values:
            steps = (num_values - cur_index) // (available_bins - 1)
            start_value = values[cur_index]
            end_index = cur_index + steps
            for end_index in range(cur_index + steps, num_values):
                if values[end_index] != start_value:
                    break
            cur_index = end_index
            available_bins -= 1
            bins.append(values[cur_index] if cur_index < num_values else values[-1])
        bins.sort()
        assert len(bins) == self.num_bins + 1
        self.bins = np.array(bins)

    def _create_sample_probs(self, values: list[float], num_sample_per_bin: int = 100) -> None:
        def normalize(arr: np.ndarray) -> np.ndarray:
            return arr / arr.sum()

        assert self.num_bins is not None and self.bins is not None
        bin_values: list[list[float]] = [[] for _ in range(self.num_bins)]
        for value in values:
            bin_values[self.get_bin_index(value)].append(value)
        bin_top_items: list[list[tuple[float, int]]] = [Counter(bin_values[i]).most_common(num_sample_per_bin) for i in range(self.num_bins)]
        self.bin_values = [np.array([value for value, _ in bin]) for bin in bin_top_items]
        self.bin_probs = [normalize(np.array([count for _, count in bin])) for bin in bin_top_items]
        assert len(self.bin_values) == self.num_bins
        assert len(self.bin_probs) == self.num_bins

    def sample(self, bin_index: int) -> float:
        """Sample from bin index."""
        assert 0 <= bin_index <= self.num_bins
        sample_value = np.random.choice(
            a=self.bin_values[bin_index],
            p=self.bin_probs[bin_index],
            replace=True,
        )
        return sample_value

    def get_bin_index(self, value: float) -> int:
        """Get bin index from value."""
        # normally return value in [1, self.num_bins - 1]
        # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        index: int = np.digitize(value, self.bins, right=True)  # type: ignore
        index = index - 1
        index = max(0, index)  # trim value < bins[0]
        index = min(index, self.num_bins - 1)  # trim value > bins[-1]
        return index


def _test_bin_converter() -> None:
    import logging

    from tqdm import tqdm

    num_values = 1000000
    values: list[float] = [np.random.randint(0, 100) for _ in range(num_values)]
    bin_converter = BinConverter()
    num_bins = 30

    logging.info("created bins:")
    bin_converter.init(values, num_bins=num_bins)
    logging.info(bin_converter.bins.tolist())
    logging.info("testing")
    for value in tqdm(values[:10000], desc="testing bin converter"):
        index = bin_converter.get_bin_index(value)
        sampled_value = bin_converter.sample(index)
        assert index == bin_converter.get_bin_index(sampled_value)

    for value in [-1, 0, 1, 2, 3, 4, 10, 20, 90, 99, 100, 101, 200]:
        logging.info(f"bin index of {value}: {bin_converter.get_bin_index(value)}")

    for index in [0, 1, 2, 10, 20, num_bins - 2, num_bins - 1]:
        sampled_value = bin_converter.sample(index)
        logging.info(f"sample index {index} -> {sampled_value}")


if __name__ == "__main__":
    _test_bin_converter()
