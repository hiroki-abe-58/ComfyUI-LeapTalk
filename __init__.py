"""ComfyUI-LeapTalk: unofficial ComfyUI integration of LeapTalk (portrait + speech -> talking-head video).

Importing this package loads no CUDA, no model and no upstream code; generation runs in a separate
runtime process registered by the administrator (see README.md).
"""

# ComfyUI imports this file as a package. pytest also imports it, as a bare module without a parent
# package, while collecting the repository root; the relative import is only meaningful in the first case.
if __package__:
    from .leaptalk_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
