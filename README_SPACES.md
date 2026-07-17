---
title: Neural Style Transfer
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app_streamlit.py
pinned: false
license: mit
---

# Neural Style Transfer

A neural style transfer web app powered by PyTorch AdaIN (Adaptive Instance Normalization).

## Features
- Upload content and style images
- Apply artistic style transfer
- Adjust alpha for blend strength
- Download stylized results

## How to Use
1. Upload a content image (the image you want to style)
2. Upload a style image (the artistic style to apply)
3. Adjust the alpha slider if needed
4. Click "Transfer Style"
5. Download your result

## Model
Uses a pre-trained VGG encoder and learned decoder for fast neural style transfer.
