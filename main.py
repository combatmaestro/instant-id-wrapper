from fastapi import FastAPI
import traceback
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import BytesIO
from PIL import Image
import torch
import base64
import requests
import numpy as np
import cv2

from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from transformers import CLIPImageProcessor

from ip_adapter.ip_adapter import IPAdapterXL

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class FaceSwapRequest(BaseModel):
    source_url: str
    target_url: str
    prompt: str = Field(default="best quality, high quality")

@app.post("/swap-face")
async def swap_face(request: FaceSwapRequest):
    try:
        src_img = Image.open(BytesIO(requests.get(request.source_url).content)).convert("RGB")
        tgt_img = Image.open(BytesIO(requests.get(request.target_url).content)).convert("RGB")

        faces = face_analysis.get(np.array(src_img))
        if not faces:
            return {"status": "error", "message": "No face detected in source image."}

        face = faces[0]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_np = np.array(src_img)[y1:y2, x1:x2]

        if face_np.size == 0:
            return {"status": "error", "message": "Face crop is empty."}

        face_img = Image.fromarray(face_np)

        images = ip_adapter.generate(
            pil_image=tgt_img,
            face_image=None,
            clip_image_embeds=torch.tensor(face.embedding).unsqueeze(0).to(device, dtype=torch.float16),
            prompt=request.prompt,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_samples=1,
            num_inference_steps=40,
            seed=42
        )

        from fastapi.responses import StreamingResponse
        output_buffer = BytesIO()
        images[0].save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {"status": "error", "message": tb}

@app.get("/")
def root():
    return {"message": "InstantID API running with identity embedding face swap."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
