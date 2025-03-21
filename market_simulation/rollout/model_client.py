from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

import ray
import requests

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@ray.remote
def send_query(url: str, state: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Send query to serving model."""
    pdata = pickle.dumps(state)
    response = requests.post(url, data=pdata)
    output = pickle.loads(response.content)
    return output


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
    """Test the ModelClient's ability to communicate with the model server and validate predictions.

    This function tests the ModelClient by:
    1. Loading validation samples and their expected outputs from disk
    2. Creating a ModelClient instance using configuration parameters
    3. Sending each sample to the model server through the client
    4. Validating that the returned predictions match the expected outputs

    The test asserts both that the expected labels match the loaded label data
    and that the model predictions match the expected outputs, failing if
    any discrepancies are found.

    Prerequisites:
        - Model server must be running and accessible at the configured IP and port
        - Model should be configured with minimal temperature to ensure deterministic outputs
          (e.g., export MODEL_SERVING__TEMPERATURE=0.00001 && bash scripts/start-model-serving.sh)
    """
    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm

    from market_simulation.conf import C
    from market_simulation.utils import pkl_utils

    data_path = Path(C.directory.input_root_dir) / "validation-samples/valid-00-00000000-0-32.zstd"
    label_path = Path(C.directory.input_root_dir) / "validation-samples/valid-00-00000000-0-32.output.csv"
    assert data_path.exists(), f"Data path {data_path} does not exist."
    label_data = pd.read_csv(label_path)
    logging.info(f"Label data shape: {label_data.shape}, head: \n{label_data.head()}")
    samples = pkl_utils.load_pkl_zstd(data_path)[: len(label_data)]
    logging.info(f"Loaded {len(samples)} samples from {data_path}")
    client = ModelClient(
        model_name=C.model_serving.model_name,
        ip=C.model_serving.ip,
        port=C.model_serving.port,
    )
    for i, (features, label) in tqdm(enumerate(samples), total=len(samples), desc="Testing model client"):
        outputs = client.get_prediction(features)
        assert outputs.size == 1
        output = outputs[0]
        assert label[-1] == label_data.iloc[i, 0], f"label: {output} != {label_data.iloc[i, 0]}"
        assert output == label_data.iloc[i, 1], f"pred: {output} != {label_data.iloc[i, 1]}"


if __name__ == "__main__":
    test_model_client()
