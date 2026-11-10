import os
import sys

import torch

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.models.adapter_base import estimate_state_bytes


def test_tensor_and_nested_cache_bytes():
    assert estimate_state_bytes(torch.zeros(2, 3)) == 2 * 3 * 4

    class LayerCache:
        def __init__(self):
            self.keys = torch.zeros(2, 2)
            self.values = torch.zeros(3, 2)
            self.is_initialized = True

    class NestedCache:
        def __init__(self):
            self.layers = [LayerCache(), LayerCache()]

    assert estimate_state_bytes(NestedCache()) > 0


def test_dynamic_cache_bytes_supported():
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    cache.update(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4), 0)
    assert estimate_state_bytes(cache) > 0
