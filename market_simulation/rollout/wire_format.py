"""Fixed-size int32 wire format for order-model requests and responses."""

from collections.abc import AsyncIterable

import numpy as np
import numpy.typing as npt

WIRE_DTYPE = np.dtype("<i4")


class PayloadTooLarge(ValueError):
    """Raised when a streamed request exceeds the fixed wire size."""


async def read_fixed_body(chunks: AsyncIterable[bytes], expected_size: int) -> bytes:
    """Read a request body without buffering more than the expected size."""
    body = bytearray()
    async for chunk in chunks:
        if len(body) + len(chunk) > expected_size:
            raise PayloadTooLarge("Order-model request is too large")
        body.extend(chunk)
    if len(body) != expected_size:
        raise ValueError("Invalid int32 payload length")
    return bytes(body)


def encode_int32(values: npt.NDArray[np.int32], expected_elements: int) -> bytes:
    """Encode exactly the expected number of int32 values in little-endian order."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.int32) or array.size != expected_elements:
        raise ValueError("Expected a fixed-size int32 array")
    return array.astype(WIRE_DTYPE, copy=False).tobytes(order="C")


def decode_int32(payload: bytes, expected_elements: int) -> npt.NDArray[np.int32]:
    """Decode a fixed-size int32 payload without object deserialization."""
    if len(payload) != expected_elements * WIRE_DTYPE.itemsize:
        raise ValueError("Invalid int32 payload length")
    return np.frombuffer(payload, dtype=WIRE_DTYPE).astype(np.int32, copy=True)
