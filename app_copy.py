import streamlit as st
from PIL import Image
import os
import torch

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("🧠 Image Caption Generator (Local Models Only)")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
BLIP_PATH = "models/blip"
VIT_PATH = "models/vit_gpt2"

# -------------------------
# Sidebar
# -------------------------
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["BLIP (Salesforce)", "ViT-GPT2 (nlpconnect)"]
)

# -------------------------
# Helper: Check model exists
# -------------------------
def is_model_available(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

# -------------------------
# Download Functions
# -------------------------
def download_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor.save_pretrained(BLIP_PATH)
    model.save_pretrained(BLIP_PATH)

def download_vit():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    model.save_pretrained(VIT_PATH)
    processor.save_pretrained(VIT_PATH)
    tokenizer.save_pretrained(VIT_PATH)

# -------------------------
# Load Models (cached)
# -------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_PATH, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH, local_files_only=True).to(device)
    return processor, model

@st.cache_resource
def load_vit():
    model = VisionEncoderDecoderModel.from_pretrained(VIT_PATH, local_files_only=True).to(device)
    processor = ViTImageProcessor.from_pretrained(VIT_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(VIT_PATH, local_files_only=True)
    return model, processor, tokenizer

# -------------------------
# MODEL CHECK (TOP LOADER)
# -------------------------
with st.spinner("🔍 Checking model availability..."):
    if model_choice == "BLIP (Salesforce)":
        model_available = is_model_available(BLIP_PATH)
    else:
        model_available = is_model_available(VIT_PATH)

# -------------------------
# If NOT available
# -------------------------
if not model_available:
    st.warning("⚠️ Model not found locally!")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬇️ Download Model"):
            with st.spinner("Downloading model... please wait ⏳"):
                os.makedirs(BLIP_PATH if model_choice.startswith("BLIP") else VIT_PATH, exist_ok=True)

                if model_choice.startswith("BLIP"):
                    download_blip()
                else:
                    download_vit()

            st.success("✅ Model downloaded! Reloading...")
            st.rerun()

    with col2:
        if st.button("❌ Cancel"):
            st.image("assets/no_model.png", caption="No model loaded 😅")
            st.stop()

    st.stop()

# -------------------------
# If AVAILABLE → Load
# -------------------------
with st.spinner("🚀 Loading model..."):
    if model_choice == "BLIP (Salesforce)":
        processor, model = load_blip()
    else:
        model, processor, tokenizer = load_vit()

st.success(f"✅ {model_choice} loaded successfully")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):

        with st.spinner("🧠 Generating caption..."):

            if model_choice == "BLIP (Salesforce)":
                inputs = processor(image, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_new_tokens=30)
                caption = processor.decode(output[0], skip_special_tokens=True)

            else:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                output_ids = model.generate(pixel_values, max_length=30)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        st.success("✨ Caption Generated")
        st.write(f"**📝 {caption}**")
