import torch
from .utils import tensor2rgba, tensor2mask, combine, apply_color_correction, PasteByMask

class CombineAndPasteOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoded_vae": ("IMAGE",),
                "Original_Image": ("IMAGE",),
                "Cut_Image": ("IMAGE",),
                "Cut_Mask": ("IMAGE",),
                "region": ("IMAGE",),
                "color_xfer_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "op": (["union (max)", "intersection (min)", "difference", "multiply", "multiply_alpha", "add", "greater_or_equal", "greater"],),
                "clamp_result": (["yes", "no"],),
                "round_result": (["no", "yes"],),
                "resize_behavior": (["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"],),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("FinalOut", )
    FUNCTION = "com_paste_op"
    CATEGORY = "I2I slim"

    def com_paste_op(self, decoded_vae, Original_Image, Cut_Image, Cut_Mask, region, color_xfer_factor, op, clamp_result, round_result, resize_behavior, mask_mapping_optional = None):

        Combined_Decoded = combine(decoded_vae, Cut_Mask, op, clamp_result, round_result)

        Combined_Originals = combine(Cut_Image, Cut_Mask, op, clamp_result, round_result)

        Cx_Decoded = apply_color_correction(Combined_Decoded, Combined_Originals, color_xfer_factor)

        Cx_Decode_Mask = combine(Cx_Decoded, Cut_Mask, op, clamp_result, round_result)

        FinalOut = PasteByMask(Original_Image, Cx_Decode_Mask, region, resize_behavior, mask_mapping_optional)

        return (FinalOut, )
