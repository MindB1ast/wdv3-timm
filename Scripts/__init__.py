# pylint: disable=invalid-name  # Preserve the legacy public import path.
"""Public notebook entry points for regional image tagging."""

from importlib import import_module

__all__ = ["ScriptOptions", "BatchTagging", "view_image_results", "compare_tagging_methods"]


def __getattr__(name):
    """Keep notebook/chart dependencies out of inference-only integrations."""
    modules = {"ScriptOptions": ".configs", "BatchTagging": ".main",
               "view_image_results": ".visualization",
               "compare_tagging_methods": ".analitycs"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
