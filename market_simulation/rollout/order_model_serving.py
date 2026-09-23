from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import torch
from ray import serve
from starlette.responses import Response

from market_simulation.conf import C
from market_simulation.models.order_model import OrderModel
from market_simulation.rollout.wire_format import PayloadTooLargeError, decode_int32, encode_int32, read_fixed_body


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": C.model_serving.num_gpus, "num_cpus": C.model_serving.num_cpus},
)
class OrderModelServing:
    """Serving Model with ray serve."""

    def __init__(self) -> None:
        self.model: OrderModel = self._load_model().cuda().eval()
        self.temperature = C.model_serving.temperature
        logging.info(f"Order model initialized, with temperature: {self.temperature}.")

    def _load_model(self) -> OrderModel:
        repo_id = C.model_serving.repo_id
        order_model = OrderModel.from_pretrained(C.model_serving.repo_id)
        logging.info(f"Loaded model from {repo_id}.")
        logging.info(f"Model configs: {order_model.num_layers}, {order_model.emb_dim}, {order_model.num_heads}")
        if C.model_serving.fp16:
            order_model.half()
            logging.info("Model converted to half precision.")
        return order_model

    @serve.batch(max_batch_size=C.model_serving.max_batch_size)  # type: ignore
    async def batch_inference(self, requests: list[npt.NDArray[np.int32]]) -> list[bytes]:
        """Batch inference."""
        batch_size = len(requests)
        input_tensor = torch.from_numpy(np.asarray(requests)).cuda()
        input_tensor = input_tensor.reshape((batch_size, C.order_model.seq_len, C.order_model.token_dim))
        logging.info(f"batch size: {batch_size}, input shape: {input_tensor.shape}")
        with torch.no_grad():
            output_tensor: np.ndarray = self.model.sample(input_tensor, self.temperature).int().cpu().reshape((batch_size, -1)).numpy()
        logging.info(f"output shape: {output_tensor.shape}")

        results: list[bytes] = []
        for i in range(batch_size):
            output = output_tensor[i]
            results.append(encode_int32(output, 1))
        return results

    async def __call__(self, request) -> Response:  # noqa: ANN001
        """Handle request."""
        expected_elements = C.order_model.seq_len * C.order_model.token_dim
        expected_bytes = expected_elements * np.dtype(np.int32).itemsize
        if request.headers.get("content-type") != "application/octet-stream":
            return Response("Expected application/octet-stream", status_code=415)
        try:
            request_bytes = await read_fixed_body(request.stream(), expected_bytes)
            arr = decode_int32(request_bytes, expected_elements)
        except PayloadTooLargeError as error:
            return Response(str(error), status_code=413)
        except ValueError as error:
            return Response(str(error), status_code=400)
        result = await self.batch_inference(arr)  # type: ignore
        return Response(result, media_type="application/octet-stream")


order_model_app = OrderModelServing.bind()  # type: ignore
