from .inpaint_segment import MaskToRegion2
from .combine_and_paste import CombineAndPasteOp

NODE_CLASS_MAPPINGS = {
    "gw_InpaintSegments": MaskToRegion2,
    "gw_CombineAndPaste": CombineAndPasteOp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "gw_InpaintSegments": "Cut (Inpaint Segments)",
    "gw_CombineAndPaste": "Paste (Combine and Paste)",
}
