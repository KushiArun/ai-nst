#!/bin/bash
# Install CPU-only PyTorch first
pip install torch==2.13.0 torchvision==0.18.0 -f https://download.pytorch.org/whl/torch_stable.html
# Then install other requirements
pip install -r requirements.txt
