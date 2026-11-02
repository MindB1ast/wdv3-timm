"""ComfyUI custom-node entry point."""
from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from importlib.util import find_spec

WEB_DIRECTORY = "./web"

if find_spec("comfy_api") is not None:
    from .comfy_ui_nodes import NODE_CLASSES
    for node in NODE_CLASSES:
        schema = node.GET_SCHEMA()
        NODE_CLASS_MAPPINGS[schema.node_id] = node
        NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
