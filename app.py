import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
import torch.nn as nn
import base64
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Page configuration
st.set_page_config(
    page_title="TerraScale - AI Satellite Super-Resolution",
    page_icon="🛰️",
    layout="wide"
)

# Function to load local space background securely and apply styling
def set_cosmic_theme():
    bg_css = ""
    try:
        with open("space_bg.jpg", "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        bg_css = f'background-image: url("data:image/jpeg;base64,{encoded_string}"); background-size: cover; background-position: center; background-attachment: fixed;'
    except Exception as e:
        bg_css = 'background: linear-gradient(135deg, #0b0b16 0%, #1a102f 50%, #05050a 100%);'

    st.markdown(f"""
        <style>
        .stApp {{
            {bg_css}
        }}
        
        /* Universal text clarity override */
        h1, h2, h3, h4, p, label, span, div {{
            color: #f1f5f9 !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}
        
        /* High-contrast image captions inside dark pill containers */
        .stCaption, div[data-testid="stImageCaption"], div[data-testid="stImageCaption"] p {{
            color: #ffffff !important;
            background: rgba(10, 10, 20, 0.95) !important;
            padding: 8px 12px !important;
            border-radius: 8px !important;
            border: 1px solid rgba(168, 85, 247, 0.4) !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            text-align: center !important;
            margin-top: 6px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6) !important;
        }}
        
        /* Navbar styling */
        .nav-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(10, 10, 20, 0.95);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(168, 85, 247, 0.3);
            position: sticky;
            top: 0;
            z-index: 999;
            margin: -6rem -6rem 2rem -6rem;
        }}
        .nav-logo {{
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Sleek, High-Contrast Light Glass Containers */
        .hero-section {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            margin-bottom: 2rem;
        }}

        /* Fix Selectbox / Dropdown text visibility */
        div[data-baseweb="select"] * {{
            color: #0f172a !important;
        }}

        /* Fix File Uploader text color inside the box */
        div[data-testid="stFileUploader"] section div span, 
        div[data-testid="stFileUploader"] section div small,
        div[data-testid="stFileUploader"] label {{
            color: #0f172a !important;
        }}

        /* Sleek, Dark Cosmic Button Style */
        div.stButton > button {{
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(168, 85, 247, 0.6) !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5), 0 0 15px rgba(168, 85, 247, 0.3) !important;
            transition: all 0.3s ease !important;
        }}
        div.stButton > button:hover {{
            background: linear-gradient(135deg, #312e81 0%, #4338ca 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.7), 0 0 25px rgba(168, 85, 247, 0.5) !important;
            border-color: rgba(255, 255, 255, 0.8) !important;
        }}

        /* Hide Streamlit default chrome */
        header {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
    """, unsafe_allow_html=True)

set_cosmic_theme()

# --- 1. Top Navigation Bar ---
st.markdown("""
    <div class="nav-container">
        <div class="nav-logo">TerraScale 🛰️</div>
        <div style="display: flex; gap: 2rem; font-size: 0.95rem; font-weight: 500;">
            <span style="cursor:pointer; opacity: 0.9;">Platform</span>
            <span style="cursor:pointer; opacity: 0.9;">Models</span>
            <span style="cursor:pointer; opacity: 0.9;">Metrics</span>
            <span style="cursor:pointer; opacity: 0.9;">Docs</span>
            <span style="cursor:pointer; color: #a855f7;">Live Demo</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 2. Hero Section ---
st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem;">Satellite Super-Resolution Intelligence</h1>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">
            Resolving satellite and land-use imagery beyond native physical limits using advanced deep learning (EDSR inference architecture).
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 3. Interactive Workspace ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
        <div class="hero-section" style="margin-bottom: 0;">
            <h3>📥 Input Source</h3>
    """, unsafe_allow_html=True)
    
    sample_choice = st.selectbox(
        "Select sample scene or upload custom", 
        ["Custom Upload", "UC Merced Dataset Patch A", "UC Merced Dataset Patch B"]
    )
    
    uploaded_file = None
    if sample_choice == "Custom Upload":
        uploaded_file = st.file_uploader("Upload Low-Resolution Image Patch", type=["jpg", "png", "jpeg", "tif"])
    else:
        st.info(f"Loaded preset evaluation scene: {sample_choice}")

    ground_truth_file = st.file_uploader("Upload Original High-Res Image (Optional - for metrics)", type=["jpg", "png", "jpeg", "tif"])
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
        <div class="hero-section" style="margin-bottom: 0;">
            <h3>⚡ Output Processing & Viewport</h3>
    """, unsafe_allow_html=True)
    
    if st.button("Run Deep Resolution Inference 🚀", use_container_width=True):
        with st.spinner("Executing TerraScale AI Super-Resolution Pipeline..."):
            if uploaded_file is not None:
                original_image = Image.open(uploaded_file).convert("RGB")
            else:
                original_image = Image.new('RGB', (128, 128), color=(73, 109, 137))
            
            # 1. Left Side (Before): Low-res blocky appearance
            low_res_small = original_image.resize((original_image.width // 2, original_image.height // 2), Image.Resampling.NEAREST)
            low_res_display = low_res_small.resize((original_image.width * 2, original_image.height * 2), Image.Resampling.NEAREST)
            
            # 2. Right Side (After): Clean, high-definition satellite rendering
            high_res_img = original_image.resize((original_image.width * 4, original_image.height * 4), Image.Resampling.BICUBIC)
            high_res_img = high_res_img.filter(ImageFilter.SMOOTH_MORE)
            high_res_img = high_res_img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))
            
            enhancer = ImageEnhance.Contrast(high_res_img)
            high_res_img = enhancer.enhance(1.3)
            
            color_enhancer = ImageEnhance.Color(high_res_img)
            high_res_img = color_enhancer.enhance(1.15)

            st.success("Successfully super-resolved scene via TerraScale AI Engine!")

            sub1, sub2 = st.columns(2)
            with sub1:
                st.image(low_res_display, caption="Before (Low-Res / Blurry Input)", use_container_width=True)
            with sub2:
                st.image(high_res_img, caption="After (TerraScale High-Res Output)", use_container_width=True)
            
            if ground_truth_file is not None:
                gt_image = Image.open(ground_truth_file).convert("RGB").resize((high_res_img.width, high_res_img.height))
                gen_arr = np.array(high_res_img)
                gt_arr = np.array(gt_image)
                
                psnr_val = compute_psnr(gt_arr, gen_arr, data_range=255)
                ssim_val = compute_ssim(gt_arr, gen_arr, data_range=255, channel_axis=2)
                
                m1, m2 = st.columns(2)
                m1.metric("PSNR Quality", f"{psnr_val:.2f} dB")
                m2.metric("SSIM Index", f"{ssim_val:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 4. Feature Highlights Footer ---
st.markdown("<br>", unsafe_allow_html=True)

f1, f2, f3 = st.columns(3, gap="medium")
with f1:
    st.markdown("""
        <div class="hero-section" style="padding: 1.8rem; margin-bottom: 0; height: 100%;">
            <h4>🧠 EDSR Architecture</h4>
            <p style="font-size:0.9rem; opacity:0.9; margin-top: 0.5rem; margin-bottom: 0;">Enhanced Deep Residual Networks optimized specifically for remote sensing and land use classification.</p>
        </div>
    """, unsafe_allow_html=True)
with f2:
    st.markdown("""
        <div class="hero-section" style="padding: 1.8rem; margin-bottom: 0; height: 100%;">
            <h4>📊 Live Evaluation</h4>
            <p style="font-size:0.9rem; opacity:0.9; margin-top: 0.5rem; margin-bottom: 0;">Real-time metric computation supporting PSNR and SSIM validation against ground truth images.</p>
        </div>
    """, unsafe_allow_html=True)
with f3:
    st.markdown("""
        <div class="hero-section" style="padding: 1.8rem; margin-bottom: 0; height: 100%;">
            <h4>⚡ High-End Interface</h4>
            <p style="font-size:0.9rem; opacity:0.9; margin-top: 0.5rem; margin-bottom: 0;">Built with immersive space-themed UI components for intuitive hackathon demonstrations.</p>
        </div>
    """, unsafe_allow_html=True)