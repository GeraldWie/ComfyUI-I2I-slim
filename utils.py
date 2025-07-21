import torch
import torch.nn.functional as F
import math
from torchvision.ops import masks_to_boxes
from PIL import Image
import numpy as np
from typing import Tuple
import torchvision.transforms.functional as TF
# Utility functions for image tensor conversion

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t
    
def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensors2common(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) < len(t2s):
        t1 = t1.unsqueeze(3)
    elif len(t1s) > len(t2s):
        t2 = t2.unsqueeze(3)

    if len(t1.size()) == 3:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1)
    else:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1, 1)

    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) > 3 and t1s[3] < t2s[3]:
        return tensor2batch(t1, t2s), t2
    elif len(t1s) > 3 and t1s[3] > t2s[3]:
        return t1, tensor2batch(t2, t1s)
    else:
        return t1, t2

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to Tensor
def pil2tensor_stacked(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    elif isinstance(image, torch.Tensor):
        return image
    else:
        raise ValueError(f"Unexpected datatype for input to 'pil2tensor_stacked'. Expected a PIL Image or tensor, but received type: {type(image)}")

def tensor2batch(t: torch.Tensor, bs: torch.Size) -> torch.Tensor:
    if len(t.size()) < len(bs):
        t = t.unsqueeze(3)
    if t.size()[0] < bs[0]:
        t.repeat(bs[0], 1, 1, 1)
    dim = bs[3]
    if dim == 1:
        return tensor2mask(t)
    elif dim == 3:
        return tensor2rgb(t)
    elif dim == 4:
        return tensor2rgba(t)


def combine(image1, image2, op, clamp_result, round_result):
    image1, image2 = tensors2common(image1, image2)

    if op == "union (max)":
        result = torch.max(image1, image2)
    elif op == "intersection (min)":
        result = torch.min(image1, image2)
    elif op == "difference":
        result = image1 - image2
    elif op == "multiply":
        result = image1 * image2
    elif op == "multiply_alpha":
        image1 = tensor2rgba(image1)
        image2 = tensor2mask(image2)
        result = torch.cat((image1[:, :, :, :3], (image1[:, :, :, 3] * image2).unsqueeze(3)), dim=3)
    elif op == "add":
        result = image1 + image2
    elif op == "greater_or_equal":
        result = torch.where(image1 >= image2, 1., 0.)
    elif op == "greater":
        result = torch.where(image1 > image2, 1., 0.)

    if clamp_result == "yes":
        result = torch.min(torch.max(result, torch.tensor(0.)), torch.tensor(1.))
    if round_result == "yes":
        result = torch.round(result)

    return result

def apply_color_correction(target_image, source_image, factor=1):

    if not isinstance(source_image, (torch.Tensor, Image.Image)):
        raise ValueError(f"Unexpected datatype for 'source_image' at method start. Expected a tensor or PIL Image, but received type: {type(source_image)}")
    
    # Ensure source_image is a tensor
    if isinstance(source_image, Image.Image):  # Check if it's a PIL Image
        source_image = pil2tensor_stacked(source_image)  # Convert it to tensor

    if not isinstance(source_image, (torch.Tensor, Image.Image)):
        raise ValueError(f"Unexpected datatype for 'source_image'. Expected a tensor or PIL Image, but received type: {type(source_image)}")

    # Get the batch size
    batch_size = source_image.shape[0]
    output_images = []

    for i in range(batch_size):
        # Convert the source and target images to NumPy arrays for the i-th image in the batch
        source_numpy = source_image[i, ...].numpy()
        target_numpy = target_image[i, ...].numpy()

        # Convert to float32
        source_numpy = source_numpy.astype(np.float32)
        target_numpy = target_numpy.astype(np.float32)

        # If the images have an alpha channel, remove it for the color transformations
        if source_numpy.shape[-1] == 4:
            source_numpy = source_numpy[..., :3]
        if target_numpy.shape[-1] == 4:
            target_numpy = target_numpy[..., :3]

        # Compute the mean and standard deviation of the color channels for both images
        target_mean, target_std = np.mean(source_numpy, axis=(0, 1)), np.std(source_numpy, axis=(0, 1))
        source_mean, source_std = np.mean(target_numpy, axis=(0, 1)), np.std(target_numpy, axis=(0, 1))

        adjusted_source_mean = target_mean + factor * (target_mean - source_mean)
        adjusted_source_std = target_std + factor * (target_std - source_std)

        # Normalize the target image (zero mean and unit variance)
        target_norm = (target_numpy - target_mean) / target_std

        # Scale and shift the normalized target image to match the exaggerated source image statistics
        matched_rgb = target_norm * adjusted_source_std + adjusted_source_mean

        # Clip values to [0, 1] and convert to PIL Image
        img = Image.fromarray(np.clip(matched_rgb * 255, 0, 255).astype('uint8'), 'RGB')

        # Convert the PIL Image to a tensor and append to the list
        img_tensor = pil2tensor_stacked(img)
        output_images.append(img_tensor)

    # Stack the list of tensors to get the batch of corrected images
    stacked_images = torch.stack(output_images)

    return stacked_images

def CutByMask(image, mask, force_resize_width, force_resize_height, mask_mapping_optional):

    if len(image.shape) < 4:
        C = 1
    else:
        C = image.shape[3]
    
    # We operate on RGBA to keep the code clean and then convert back after
    image = tensor2rgba(image)
    mask = tensor2mask(mask)

    if mask_mapping_optional is not None:
        mask_mapping_optional = mask_mapping_optional.long()
        image = image[mask_mapping_optional]

    # Scale the mask to match the image size if it isn't
    B, H, W, _ = image.shape
    mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
    
    MB, _, _ = mask.shape

    if MB < B:
        assert(B % MB == 0)
        mask = mask.repeat(B // MB, 1, 1)

    # Masks to boxes
    is_empty = ~torch.gt(torch.max(torch.reshape(mask, [B, H * W]), dim=1).values, 0.)
    mask[is_empty,0,0] = 1.
    boxes = masks_to_boxes(mask)
    mask[is_empty,0,0] = 0.

    min_x = boxes[:,0]
    min_y = boxes[:,1]
    max_x = boxes[:,2]
    max_y = boxes[:,3]

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    use_width = int(torch.max(width).item())
    use_height = int(torch.max(height).item())

    if force_resize_width > 0:
        use_width = force_resize_width

    if force_resize_height > 0:
        use_height = force_resize_height

    print("use_width: ", use_width)
    print("use_height: ", use_height)

    alpha_mask = torch.ones((B, H, W, 4))
    alpha_mask[:,:,:,3] = mask

    image = image * alpha_mask

    result = torch.zeros((B, use_height, use_width, 4))
    for i in range(0, B):
        if not is_empty[i]:
            ymin = int(min_y[i].item())
            ymax = int(max_y[i].item())
            xmin = int(min_x[i].item())
            xmax = int(max_x[i].item())
            single = (image[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
            resized = F.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
            result[i] = resized[0]

    # Preserve our type unless we were previously RGB and added non-opaque alpha due to the mask size
    if C == 1:
        print("C == 1 output image shape: ", tensor2mask(result).shape)
        return tensor2mask(result)
    elif C == 3 and torch.min(result[:,:,:,3]) == 1:
        print("C == 3 output image shape: ", tensor2rgb(result).shape)
        return tensor2rgb(result)
    else:
        print("else result shape: ", result.shape)
        return result


def PasteByMask(image_base, image_to_paste, mask, resize_behavior, mask_mapping_optional):
    image_base = tensor2rgba(image_base)
    image_to_paste = tensor2rgba(image_to_paste)
    mask = tensor2mask(mask)

    # Scale the mask to be a matching size if it isn't
    B, H, W, C = image_base.shape
    MB = mask.shape[0]
    PB = image_to_paste.shape[0]
    if mask_mapping_optional is None:
        if B < PB:
            assert(PB % B == 0)
            image_base = image_base.repeat(PB // B, 1, 1, 1)
        B, H, W, C = image_base.shape
        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)
        elif B < MB:
            assert(MB % B == 0)
            image_base = image_base.repeat(MB // B, 1, 1, 1)
        if PB < B:
            assert(B % PB == 0)
            image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)
    mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
    MB, MH, MW = mask.shape

    # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
    is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
    mask[is_empty,0,0] = 1.
    boxes = masks_to_boxes(mask)
    mask[is_empty,0,0] = 0.

    min_x = boxes[:,0]
    min_y = boxes[:,1]
    max_x = boxes[:,2]
    max_y = boxes[:,3]
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    target_width = max_x - min_x + 1
    target_height = max_y - min_y + 1

    result = image_base.detach().clone()

    for i in range(0, MB):
        if i >= len(image_to_paste):
            raise ValueError(f"image_to_paste does not have an entry for mask index {i}")
        if is_empty[i]:
            continue
        else:
            image_index = i
            if mask_mapping_optional is not None:
                image_index = mask_mapping_optional[i].item()
            source_size = image_to_paste.size()
            SB, SH, SW, _ = image_to_paste.shape

            # Figure out the desired size
            width = int(target_width[i].item())
            height = int(target_height[i].item())
            if resize_behavior == "keep_ratio_fill":
                target_ratio = width / height
                actual_ratio = SW / SH
                if actual_ratio > target_ratio:
                    width = int(height * actual_ratio)
                elif actual_ratio < target_ratio:
                    height = int(width / actual_ratio)
            elif resize_behavior == "keep_ratio_fit":
                target_ratio = width / height
                actual_ratio = SW / SH
                if actual_ratio > target_ratio:
                    height = int(width / actual_ratio)
                elif actual_ratio < target_ratio:
                    width = int(height * actual_ratio)
            elif resize_behavior == "source_size" or resize_behavior == "source_size_unmasked":
                width = SW
                height = SH

            # Resize the image we're pasting if needed
            resized_image = image_to_paste[i].unsqueeze(0)
            if SH != height or SW != width:
                resized_image = F.interpolate(resized_image.permute(0, 3, 1, 2), size=(height,width), mode='bicubic').permute(0, 2, 3, 1)

            pasting = torch.ones([H, W, C])
            ymid = float(mid_y[i].item())
            ymin = int(math.floor(ymid - height / 2)) + 1
            ymax = int(math.floor(ymid + height / 2)) + 1
            xmid = float(mid_x[i].item())
            xmin = int(math.floor(xmid - width / 2)) + 1
            xmax = int(math.floor(xmid + width / 2)) + 1

            _, source_ymax, source_xmax, _ = resized_image.shape
            source_ymin, source_xmin = 0, 0

            if xmin < 0:
                source_xmin = abs(xmin)
                xmin = 0
            if ymin < 0:
                source_ymin = abs(ymin)
                ymin = 0
            if xmax > W:
                source_xmax -= (xmax - W)
                xmax = W
            if ymax > H:
                source_ymax -= (ymax - H)
                ymax = H

            pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
            pasting[:, :, 3] = 1.

            pasting_alpha = torch.zeros([H, W])
            pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

            if resize_behavior == "keep_ratio_fill" or resize_behavior == "source_size_unmasked":
                # If we explicitly want to fill the area, we are ok with extending outside
                paste_mask = pasting_alpha.unsqueeze(2).repeat(1, 1, 4)
            else:
                paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)
            result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)
    return result



