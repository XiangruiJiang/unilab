"""Portable environment payloads compatible with Torch's weights-only loader.

Task terms may return NumPy arrays, but those arrays must not add arbitrary
pickle globals to a policy checkpoint: playback preflights checkpoints with
``weights_only=True``. Encode them as tagged CPU tensors and restore their NumPy
type at the explicit environment boundary. Simulator objects are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

_NUMPY_ARRAY_KEY = "__unilab_numpy_array__"


def encode_environment_state(value: Any) -> Any:
    """Detach plain containers, numeric arrays and tensors into safe CPU data."""
    if isinstance(value, np.ndarray):
        return {_NUMPY_ARRAY_KEY: torch.from_numpy(value.copy())}
    if isinstance(value, np.generic):
        return encode_environment_state(value.item())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value) or _NUMPY_ARRAY_KEY in value:
            raise ValueError(
                "Environment checkpoint keys must be strings without reserved array tags"
            )
        return {key: encode_environment_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(encode_environment_state(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"Unsupported environment checkpoint value: {type(value).__name__}")


def decode_environment_state(value: Any) -> Any:
    """Restore NumPy arrays and CPU tensors even when model weights load on CUDA."""
    if isinstance(value, dict):
        if _NUMPY_ARRAY_KEY in value:
            array = value[_NUMPY_ARRAY_KEY]
            if len(value) != 1 or not isinstance(array, torch.Tensor):
                raise ValueError("Invalid NumPy array tag in environment checkpoint")
            return array.detach().cpu().numpy().copy()
        return {key: decode_environment_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(decode_environment_state(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value
