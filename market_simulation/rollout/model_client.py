from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ray
import requests

from market_simulation.conf import C
from market_simulation.rollout.wire_format import decode_int32, encode_int32

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@ray.remote
def send_query(url: str, state: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Send query to serving model."""
    expected_elements = C.order_model.seq_len * C.order_model.token_dim
    pdata = encode_int32(state, expected_elements)
    response = requests.post(url, data=pdata, headers={"Content-Type": "application/octet-stream"})
    response.raise_for_status()
    return decode_int32(response.content, 1)


class ModelClient:
    """Inference client."""

    def __init__(
        self,
        model_name: str,
        ip: str = "localhost",
        port: int = 8000,
    ) -> None:
        self.url: str = f"http://{ip}:{port}/{model_name}"
        logging.info(f"Created client for model: {self.url}")

    def get_prediction(self, state: np.ndarray) -> npt.NDArray[np.int32]:
        """Get predictin from serving model."""
        if not ray.is_initialized():
            ray.init()
        output: npt.NDArray[np.int32] = ray.get(send_query.remote(self.url, state))
        output = output.reshape(-1)
        return output


def test_model_client() -> None:
    """Test the ModelClient's ability to communicate with the model server.

    This function tests the ModelClient by:
    1. Loading validation samples from disk
    2. Creating a ModelClient instance using configuration parameters
    3. Sending each sample to the model server through the client
    4. Validating that the server returns one prediction per sample

    Prerequisites:
        - Model server must be running and accessible at the configured IP and port
    """
    from pathlib import Path

    from tqdm import tqdm

    from market_simulation.conf import C
    from market_simulation.utils import pkl_utils

    data_path = Path(C.directory.input_root_dir) / "validation-samples/valid-00-00000000-0-32.zstd"
    assert data_path.exists(), f"Data path {data_path} does not exist."
    samples = pkl_utils.load_pkl_zstd(data_path)[:1]
    logging.info(f"Loaded {len(samples)} samples from {data_path}")
    client = ModelClient(
        model_name=C.model_serving.model_name,
        ip=C.model_serving.ip,
        port=C.model_serving.port,
    )
    for features, _ in tqdm(samples, total=len(samples), desc="Testing model client"):
        outputs = client.get_prediction(features)
        assert outputs.size == 1
        assert 0 <= outputs[0] < 49152


if __name__ == "__main__":
    test_model_client()
