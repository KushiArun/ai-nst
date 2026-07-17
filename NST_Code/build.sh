#!/bin/bash
# Install CPU-only PyTorch first
pip install torch==2.0.1 torchvision==0.15.2 -f https://download.pytorch.org/whl/torch_stable.html
# Then install other requirements
pip install -r requirements.txt
