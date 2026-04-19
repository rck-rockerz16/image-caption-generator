import os
import argparse
from PIL import Image
import torch


from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
# CONFIG
device = "cuda" if torch.cuda.is_available() else "cpu"

BLIP_PATH = "models/blip"
VIT_PATH = "models/vit_gpt2"

# Helper Functions
def is_model_available(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0


def download_blip():
    print("⬇️ Downloading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    os.makedirs(BLIP_PATH, exist_ok=True)
    processor.save_pretrained(BLIP_PATH)
    model.save_pretrained(BLIP_PATH)

    print("✅ BLIP model downloaded successfully")


def download_vit():
    print("⬇️ Downloading ViT-GPT2 model...")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    os.makedirs(VIT_PATH, exist_ok=True)
    model.save_pretrained(VIT_PATH)
    processor.save_pretrained(VIT_PATH)
    tokenizer.save_pretrained(VIT_PATH)

    print("✅ ViT-GPT2 model downloaded successfully")


def load_blip():
    print("🚀 Loading BLIP model...")
    processor = BlipProcessor.from_pretrained(BLIP_PATH, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_PATH, local_files_only=True
    ).to(device)
    return processor, model


def load_vit():
    print("🚀 Loading ViT-GPT2 model...")
    model = VisionEncoderDecoderModel.from_pretrained(
        VIT_PATH, local_files_only=True
    ).to(device)
    processor = ViTImageProcessor.from_pretrained(
        VIT_PATH, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        VIT_PATH, local_files_only=True
    )
    return model, processor, tokenizer


def ensure_model(model_name):
    path = BLIP_PATH if model_name == "blip" else VIT_PATH

    print("🔍 Checking model availability...")
    if is_model_available(path):
        print("✅ Model found locally")
        return

    choice = input("⚠️ Model not found. Download now? (y/n): ").strip().lower()

    if choice != "y":
        print("❌ Exiting. No model loaded.")
        exit()

    if model_name == "blip":
        download_blip()
    else:
        download_vit()


def generate_caption(image_path, model_name):
    if not os.path.exists(image_path):
        print("❌ Invalid image path")
        return

    image = Image.open(image_path).convert("RGB")

    print("🧠 Generating caption...")

    if model_name == "blip":
        processor, model = load_blip()
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)

    else:
        model, processor, tokenizer = load_vit()
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=30)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "-" * 50)
    print("📝 Caption:")
    print(caption)
    print("-" * 50)

# MAIN
def main():
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument("--model", choices=["blip", "vit"], help="Choose model")
    parser.add_argument("--image", help="Path to image")

    args = parser.parse_args()

    # Direct mode
    if args.model and args.image:
        ensure_model(args.model)
        generate_caption(args.image, args.model)

    # Interactive mode
    else:
        print("\n🧠 Image Caption Generator")
        print("1. BLIP (Salesforce)")
        print("2. ViT-GPT2 (nlpconnect)")

        choice = input("Choose model (1/2): ").strip()

        model_name = "blip" if choice == "1" else "vit"

        ensure_model(model_name)

        image_path = input("Enter image path: ").strip()

        generate_caption(image_path, model_name)


if __name__ == "__main__":
    main()
