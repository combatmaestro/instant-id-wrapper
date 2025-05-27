from fastapi import FastAPI
import traceback
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
import base64
import requests
import numpy as np

from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from transformers import CLIPImageProcessor

from ip_adapter.ip_adapter import IPAdapterXL

# ========== FastAPI setup ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Model loading ==========
print("[INFO] Loading InstantID components...")
device = "cuda" if torch.cuda.is_available() else "cpu"

face_analysis = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_analysis.prepare(ctx_id=0)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

try:
    ip_adapter = IPAdapterXL(
        pipe,
        image_encoder_path="/workspace/instant-id-wrapper/models/controlnet/image_encoder",
        ip_ckpt="./models/ip-adapter/ip-adapter_sdxl.bin",
        device=device
    )
except Exception as e:
    print("[FATAL ERROR] IPAdapter initialization failed!")
    traceback.print_exc()
    raise e

# ========== Input model ==========
class FaceSwapRequest(BaseModel):
    source_url: str
    target_url: str
    prompt: str

# ========== Endpoint ==========
@app.post("/swap-face")
async def swap_face(request: FaceSwapRequest):
    try:
        src_response = requests.get(request.source_url)
        tgt_response = requests.get(request.target_url)

        src_img = Image.open(BytesIO(src_response.content)).convert("RGB")
        tgt_img = Image.open(BytesIO(tgt_response.content)).convert("RGB")

        faces = face_analysis.get(np.array(src_img))
        if not faces:
            return {"status": "error", "message": "No face detected in source image."}

        face = faces[0]
        print("[INFO] Detected face[0]:", face)

        if face is None or not hasattr(face, 'crop_face') or not callable(face.crop_face):
            return {"status": "error", "message": "Face object invalid or crop_face() missing."}

        cropped = face.crop_face()
        if cropped is None:
            return {"status": "error", "message": "crop_face() returned None."}

        face_img = Image.fromarray(cropped)

        images = ip_adapter.generate(
            pil_image=tgt_img,
            face_image=face_img,
            prompt=request.prompt,
            num_samples=1,
            num_inference_steps=40,
            seed=42
        )

        output_buffer = BytesIO()
        images[0].save(output_buffer, format="PNG")
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "message": "Face swapped successfully.",
            "output_base64": output_base64,
            "mime_type": "image/png"
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"status": "error", "message": tb}

@app.get("/")
def root():
    return {"message": "InstantID API running with IP-Adapter + SDXL."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
