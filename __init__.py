from .inpaint_segment import MaskToRegion
from .combine_and_paste import Combine_And_Paste_Op

NODE_CLASS_MAPPINGS = {
    "Inpaint Segments": MaskToRegion,
    "Combine and Paste": Combine_And_Paste_Op,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Inpaint Segments": "Cut (Inpaint Segments)",
    "Combine and Paste": "Paste (Combine and Paste)",
}
