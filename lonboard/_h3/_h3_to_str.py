from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def h3_to_str(h3_indices: NDArray[np.uint64]) -> NDArray[np.str_]:
    """Convert an array of H3 indices (uint64) to their hexadecimal string representations.

    Returns a numpy array of type S15 (fixed-length ASCII strings of length 15).
    """
    # Ensure input is a numpy array of uint64
    hex_chars = np.empty((h3_indices.size, 15), dtype="S1")

    # Prepare hex digits lookup
    hex_digits = np.frombuffer(b"0123456789ABCDEF", dtype="S1")

    # Vectorized digit extraction
    shifts = np.arange(14, -1, -1, dtype=np.uint64) * 4
    digit_values = ((h3_indices[:, None] >> shifts[None, :]) & 0xF).astype(np.uint8)
    hex_chars[:] = hex_digits[digit_values]

    return hex_chars.view("<S15")[:, 0]
