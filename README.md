# ComfyUI I2I-Slim

A lightweight subset of the custom nodes originally developed by [ManglerFTW](https://github.com/ManglerFTW/ComfyI2I) for performing image-to-image tasks in ComfyUI.

This slim version includes only the following nodes:

## ✂️ Inpaint Segments Node

This node segments and crops both your image and mask based on the bounding boxes of each mask region. It then resizes each cropped segment to 1024×1024 (or a custom size you define).

## 📌 Combine and Paste Node

This node takes the newly generated images (e.g., from the VAE Decode node), resizes them to match the original mask bounding boxes, and pastes them back into the original image at the correct locations.

## 🚀 Installation

1. Drop this folder into `ComfyUI/custom_nodes/`
2. Install required packages:

   ```bash
   pip install -r requirements.txt
