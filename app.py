from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import io
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download

# Configurații
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Download + Load SAM ===
checkpoint = hf_hub_download(
    repo_id="MetallicGroup/sam-vit-checkpoint",
    filename="sam_vit_h_4b8939.pth"
)
sam = sam_model_registry["vit_h"](checkpoint=checkpoint).to(DEVICE)
predictor = SamPredictor(sam)

# === Load ControlNet + Inpainting ===
controlnet = ControlNetModel.from_pretrained("lllyasviel/controlnet-inpaint", torch_dtype=torch.float16).to(DEVICE)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(DEVICE)
pipe.enable_xformers_memory_efficient_attention()

# === App ===
app = Flask(__name__)

def segment_mask(image):
    image_np = np.array(image)
    h, w, _ = image_np.shape
    predictor.set_image(image_np)
    input_box = np.array([0, int(h * 0.5), w, h])
    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
    return Image.fromarray((masks[0].astype(np.uint8) * 255)).convert("L")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        model_img = Image.open(io.BytesIO(request.files['model'].read())).convert("RGB").resize((768, 768))
        client_img = Image.open(io.BytesIO(request.files['client'].read())).convert("RGB").resize((768, 768))
        mask_img = segment_mask(client_img)

        result = pipe(
            prompt="",
            image=client_img,
            mask_image=mask_img,
            control_image=model_img,
            guidance_scale=9.0,
            num_inference_steps=40
        ).images[0]

        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        return send_file(output, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Fence AI backend running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
