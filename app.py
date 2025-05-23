import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Selectăm dispozitivul
device = "cuda" if torch.cuda.is_available() else "cpu"

# Încărcăm modelul de inpainting
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Prompturi pentru fiecare model de gard
PROMPT_MAP = {
    "MX15": "A modern vertical MX15 metal fence between concrete pillars in front of a house, realistic photo, sharp",
    "MX25": "A modern horizontal MX25 aluminum fence, realistic look, inserted between concrete pillars",
    "MX60": "A dense dark horizontal MX60 metal fence between classic concrete fence posts, photorealistic"
}

# Funcția de generare
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

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Fence AI – Înlocuire gard desenată manual")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="1️⃣ Încarcă imaginea", type="pil")
            mask_input = gr.Sketchpad(label="2️⃣ Desenează zona gardului", shape=(512, 512))
            dropdown = gr.Dropdown(["MX15", "MX25", "MX60"], label="3️⃣ Alege modelul de gard")
            generate_btn = gr.Button("🎨 Generează imaginea")
        with gr.Column():
            output = gr.Image(label="✅ Gardul AI")

    generate_btn.click(fn=generate, inputs=[image_input, mask_input, dropdown], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

