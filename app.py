import os

# Silence TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim

# =======================
# Load trained model
# =======================
MODEL_PATH = "image_denoising_autoencoder.h5"
model = load_model(MODEL_PATH, compile=False)

# =======================
# Utility Functions
# =======================
def preprocess_gray(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    return image.reshape(128, 128, 1)

def add_noise(image, noise_factor):
    noisy = image + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=image.shape
    )
    return np.clip(noisy, 0.0, 1.0)

def psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(1.0 / mse)

def interpret_psnr(value):
    if value < 15:
        return "Poor ❌"
    elif value < 25:
        return "Moderate ⚠️"
    else:
        return "Good ✅"

def interpret_ssim(value):
    if value < 0.5:
        return "Low ❌"
    elif value < 0.8:
        return "Moderate ⚠️"
    else:
        return "High ✅"

# =======================
# 🔥 POST-PROCESSING
# =======================
def enhance_clarity(image, strength=1.0):
    image_uint8 = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    sharpened = cv2.addWeighted(
        image_uint8, 1 + strength, blurred, -strength, 0
    )
    return np.clip(sharpened.astype("float32") / 255.0, 0, 1)

# =======================
# Inference Function
# =======================
def denoise_image(input_image, noise_level, mode, clarity_strength):
    if input_image is None:
        return None, None, None, "Upload an image to start."

    # -------- Grayscale --------
    if mode == "Grayscale":
        if input_image.ndim == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = input_image

        clean = preprocess_gray(gray)
        noisy = add_noise(clean, noise_level)
        denoised = model.predict(noisy.reshape(1, 128, 128, 1))[0]

        enhanced = enhance_clarity(
            denoised.reshape(128, 128),
            strength=clarity_strength
        )

        psnr_val = psnr(clean.reshape(128, 128), denoised.reshape(128, 128))
        ssim_val = ssim(
            clean.reshape(128, 128),
            denoised.reshape(128, 128),
            data_range=1.0
        )

        metrics = (
            f"PSNR: {psnr_val:.2f} dB ({interpret_psnr(psnr_val)})\n"
            f"SSIM: {ssim_val:.3f} ({interpret_ssim(ssim_val)})\n"
            f"Clarity Strength: {clarity_strength:.2f}"
        )

        return (
            clean.reshape(128, 128),
            noisy.reshape(128, 128),
            enhanced,
            metrics
        )

    # -------- RGB --------
    resized = cv2.resize(input_image, (128, 128))
    resized = resized.astype("float32") / 255.0

    noisy_rgb = add_noise(resized, noise_level)
    denoised_rgb = np.zeros_like(noisy_rgb)

    for c in range(3):
        channel = noisy_rgb[:, :, c].reshape(128, 128, 1)
        denoised_rgb[:, :, c] = model.predict(
            channel.reshape(1, 128, 128, 1)
        )[0].reshape(128, 128)

    enhanced_rgb = np.zeros_like(denoised_rgb)
    for c in range(3):
        enhanced_rgb[:, :, c] = enhance_clarity(
            denoised_rgb[:, :, c],
            strength=clarity_strength
        )

    psnr_val = psnr(resized, denoised_rgb)
    ssim_val = ssim(
        resized,
        denoised_rgb,
        channel_axis=2,
        data_range=1.0
    )

    metrics = (
        f"PSNR: {psnr_val:.2f} dB ({interpret_psnr(psnr_val)})\n"
        f"SSIM: {ssim_val:.3f} ({interpret_ssim(ssim_val)})\n"
        f"Clarity Strength: {clarity_strength:.2f}"
    )

    return resized, noisy_rgb, enhanced_rgb, metrics

# =======================
# Gradio UI
# =======================
with gr.Blocks(title="Image Denoising Autoencoder") as demo:
    gr.Markdown(
        """
        # 🧠 Image Denoising using U-Net Autoencoder  
        🎚 Sliders update output automatically
        """
    )

    with gr.Row():
        input_image = gr.Image(label="Upload Image", type="numpy", height=320)

        with gr.Column():
            noise_slider = gr.Slider(0.05, 0.5, 0.3, 0.05, label="Noise Level")
            clarity_slider = gr.Slider(0.0, 2.0, 1.0, 0.1, label="Clarity Strength")
            mode_selector = gr.Radio(
                ["Grayscale", "RGB"], value="Grayscale", label="Mode"
            )

    denoise_btn = gr.Button("Denoise Image 🚀")

    with gr.Row():
        original_out = gr.Image(label="Original", height=260)
        noisy_out = gr.Image(label="Noisy", height=260)
        denoised_out = gr.Image(label="Denoised", height=260)

    metrics_box = gr.Textbox(label="Quality Metrics", lines=4)

    inputs = [input_image, noise_slider, mode_selector, clarity_slider]
    outputs = [original_out, noisy_out, denoised_out, metrics_box]

    # 🔁 AUTO UPDATE
    noise_slider.change(denoise_image, inputs, outputs)
    clarity_slider.change(denoise_image, inputs, outputs)
    mode_selector.change(denoise_image, inputs, outputs)

    # 🔘 MANUAL BUTTON
    denoise_btn.click(denoise_image, inputs, outputs)

# =======================
# Launch App
# =======================
if __name__ == "__main__":
    demo.launch(debug=True)
