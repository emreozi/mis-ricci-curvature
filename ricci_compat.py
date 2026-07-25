"""Cross-platform compatibility helpers for GraphRicciCurvature.

GraphRicciCurvature 0.5.3.2 requests the POSIX-only ``fork``
multiprocessing context even when a single worker is selected.  The
reproducibility scripts use modest graphs, so a deterministic serial pool is
both sufficient and portable on Windows, macOS, and Linux.
"""

from __future__ import annotations

import multiprocessing as mp
import importlib


class _SerialPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, func, iterable, *args, **kwargs):
        return [func(item) for item in iterable]

    def imap(self, func, iterable, *args, **kwargs):
        return (func(item) for item in iterable)

    def imap_unordered(self, func, iterable, *args, **kwargs):
        return (func(item) for item in iterable)

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


class _SerialContext:
    def Pool(self, *args, **kwargs):
        return _SerialPool()


def install_serial_pool_patch() -> None:
    """Replace ``multiprocessing.get_context`` with a serial context once."""

    if getattr(mp.get_context, "_mis_ricci_serial_patch", False):
        return

    def _serial_get_context(method=None):
        return _SerialContext()

    _serial_get_context._mis_ricci_serial_patch = True
    mp.get_context = _serial_get_context


install_serial_pool_patch()


def clear_ricci_caches() -> None:
    """Clear graph- and alpha-dependent caches before a new computation.

    GraphRicciCurvature memoizes neighborhood distributions only by node
    index.  Reusing the module for another graph or another alpha value can
    therefore return stale neighborhoods unless these caches are cleared.
    """

    module = importlib.import_module("GraphRicciCurvature.OllivierRicci")
    module._get_single_node_neighbors_distributions.cache_clear()
    module._source_target_shortest_path.cache_clear()
