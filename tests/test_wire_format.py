"""Checks for the order-model network format and bounded request reader."""

import asyncio
import pickle
import unittest

import numpy as np

from market_simulation.rollout.wire_format import PayloadTooLarge, decode_int32, encode_int32, read_fixed_body


async def chunks(*parts: bytes):
    """Yield a request body in arbitrary network chunks."""
    for part in parts:
        yield part


def fail_if_unpickled() -> None:
    """Make an unsafe deserialization observable to the test."""
    raise AssertionError("Untrusted pickle was executed")


class PickleTrap:
    """A payload that must remain inert on the wire."""

    def __reduce__(self):
        """Return a callable that fails if deserialized."""
        return fail_if_unpickled, ()


class WireFormatTests(unittest.TestCase):
    """Exercise fixed-size int32 transport at the network boundary."""

    def test_round_trip(self) -> None:
        values = np.array([1, -2, 123456], dtype=np.int32)
        payload = encode_int32(values, 3)
        self.assertEqual(payload, b"\x01\x00\x00\x00\xfe\xff\xff\xff\x40\xe2\x01\x00")
        np.testing.assert_array_equal(decode_int32(payload, 3), values)

    def test_rejects_wrong_type_and_size(self) -> None:
        with self.assertRaises(ValueError):
            encode_int32(np.array([1], dtype=object), 1)
        with self.assertRaises(ValueError):
            encode_int32(np.array([1], dtype=np.int32), 2)
        with self.assertRaises(ValueError):
            decode_int32(b"\x00" * 3, 1)

    def test_stream_is_bounded_and_exact(self) -> None:
        self.assertEqual(asyncio.run(read_fixed_body(chunks(b"ab", b"cd"), 4)), b"abcd")
        with self.assertRaises(PayloadTooLarge):
            asyncio.run(read_fixed_body(chunks(b"abcd", b"e" * 1000000), 4))
        with self.assertRaises(ValueError):
            asyncio.run(read_fixed_body(chunks(b"abc"), 4))

    def test_pickle_payload_is_never_deserialized(self) -> None:
        payload = pickle.dumps(PickleTrap()).ljust(64, b"\x00")
        decoded = decode_int32(payload, 16)
        self.assertEqual(decoded.size, 16)


if __name__ == "__main__":
    unittest.main()
