from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
import base64
import requests
import uvicorn
import os
import numpy as np

from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from transformers import CLIPImageProcessor

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

face_analysis = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_analysis.prepare(ctx_id=0 if device == "cuda" else -1)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)
pipe.enable_model_cpu_offload()

# You can later load IP-Adapter + ControlNet

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

        face_embedding = faces[0].embedding  # To use in IP-Adapter later

        # 🛠 Replace below with actual face-swapped generation using InstantID
        output_buffer = BytesIO()
        tgt_img.save(output_buffer, format="PNG")
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "message": "Face swapped successfully (stub).",
            "output_base64": output_base64,
            "mime_type": "image/png"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"message": "InstantID API running with real model stubs."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
