import gradio as gr
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Prompt map
PROMPT_MAP = {
    "MX15": "A house with modern MX15 metal fence, vertical slats between concrete posts.",
    "MX25": "A house with MX25 horizontal metal fence, clean modern look between posts.",
    "MX60": "A house with dense horizontal MX60 metallic fence between concrete pillars."
}

# Generate image
def generate(image, mask, model_type):
    image = image.resize((768, 768))
    mask = mask.resize((768, 768)).convert("L")
    prompt = PROMPT_MAP.get(model_type, "modern metal fence")

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=8.0,
        num_inference_steps=40
    ).images[0]

    return result

# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Fence AI – Înlocuire gard cu desen manual")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="1️⃣ Încarcă poza cu gardul", type="pil")
            sketch_mask = gr.Sketchpad(label="2️⃣ Desenează zona gardului de înlocuit", shape=(512, 512))
            dropdown = gr.Dropdown(["MX15", "MX25", "MX60"], label="3️⃣ Alege modelul de gard")
            btn = gr.Button("🎨 Generează gardul AI")
        with gr.Column():
            output = gr.Image(label="✅ Rezultat")

    btn.click(fn=generate, inputs=[input_image, sketch_mask, dropdown], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
