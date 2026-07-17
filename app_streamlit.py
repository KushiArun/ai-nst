import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os

# Add NST_Code to path
import sys
sys.path.append('NST_Code')

from NST_Code.utils.models import VGGEncoder, Decoder
from NST_Code.utils.utils import adaptive_instance_normalization

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

st.title("🎨 Neural Style Transfer")
st.write("Upload content and style images to apply neural style transfer")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    """Load models (cached to avoid reloading)"""
    encoder = VGGEncoder('NST_Code/vgg_normalised.pth').to(device)
    decoder = Decoder().to(device)
    
    # Load trained decoder
    decoder.load_state_dict(
        torch.load('NST_Code/experiment/final_exp/decoder_final.pth', map_location=device)
    )
    
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    """Apply style transfer"""
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)
        
        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )
        
        stylized_feats = (
            alpha * stylized_feats +
            (1 - alpha) * content_feats
        )
        
        stylized_image = decoder(stylized_feats)
    
    return stylized_image

# Load models
try:
    encoder, decoder = load_models()
    st.success("✓ Models loaded successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# UI Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload content image", type=['png', 'jpg', 'jpeg'], key='content')
    if content_file:
        content_image = Image.open(content_file)
        st.image(content_image, use_column_width=True)

with col2:
    st.subheader("Style Image")
    style_file = st.file_uploader("Upload style image", type=['png', 'jpg', 'jpeg'], key='style')
    if style_file:
        style_image = Image.open(style_file)
        st.image(style_image, use_column_width=True)

with col3:
    st.subheader("Settings")
    alpha = st.slider("Alpha (blend strength)", 0.0, 1.0, 1.0, 0.1)
    process_button = st.button("✨ Transfer Style", use_container_width=True)

# Process
if process_button:
    if content_file is None or style_file is None:
        st.warning("Please upload both content and style images")
    else:
        with st.spinner("Processing... this may take a minute"):
            try:
                content_image = Image.open(content_file).convert('RGB')
                style_image = Image.open(style_file).convert('RGB')
                
                output = style_transfer(content_image, style_image, encoder, decoder, alpha, device)
                
                # Convert output to image
                output = output.cpu().clone()
                output = output.squeeze(0)
                output = output.clamp(0, 1)
                output_image = transforms.ToPILImage()(output)
                
                st.success("✓ Style transfer complete!")
                st.image(output_image, caption="Stylized Output", use_column_width=True)
                
                # Download button
                output_image.save("output.png")
                with open("output.png", "rb") as f:
                    st.download_button(
                        label="Download Output",
                        data=f,
                        file_name="stylized_image.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Error during style transfer: {e}")
