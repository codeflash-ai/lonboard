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

    # Vectorized hex conversion for 15 digits; avoids Python loops for performance
    n = h3_indices.size
    # Allocate output shape
    hex_chars = np.empty((n, 15), dtype="S1")
    # Prepare the hex digits lookup table
    hex_digits = np.frombuffer(b"0123456789ABCDEF", dtype="S1")

    # Compute the shifts for each digit position all at once (broadcasted)
    shifts = np.arange(14, -1, -1, dtype=np.uint64) * 4
    # Expand shifts for broadcasting against h3_indices
    bits = (h3_indices[:, None] >> shifts) & 0xF
    hex_chars[:] = hex_digits[bits]

    return hex_chars.view("<S15")[:, 0]
