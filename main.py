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
    prompt: str = Field(default="best quality, high quality")

# ========== Endpoint ==========
@app.post("/swap-face")
async def swap_face(request: FaceSwapRequest):
    try:
        src_response = requests.get(request.source_url)
        tgt_response = requests.get(request.target_url)

        src_img = Image.open(BytesIO(src_response.content)).convert("RGB")
        tgt_img = Image.open(BytesIO(tgt_response.content)).convert("RGB")
        tgt_np = np.array(tgt_img)

        faces_src = face_analysis.get(np.array(src_img))
        if not faces_src:
            return {"status": "error", "message": "No face detected in source image."}

        face_src = faces_src[0]
        print("[INFO] Detected source face[0]:", face_src)

        bbox_src = face_src.get('bbox')
        if bbox_src is None or len(bbox_src) != 4:
            return {"status": "error", "message": "Invalid or missing bounding box for face."}

        x1_src, y1_src, x2_src, y2_src = map(int, bbox_src)
        h_src, w_src = src_img.size
        x1_src, y1_src = max(0, x1_src), max(0, y1_src)
        x2_src, y2_src = min(w_src, x2_src), min(h_src, y2_src)

        if x2_src <= x1_src or y2_src <= y1_src:
            return {"status": "error", "message": "Source bounding box resulted in invalid region."}
        faces_tgt = face_analysis.get(np.array(tgt_img))
        if not faces_tgt:
            return {"status": "error", "message": "No face detected in target image."}

        face_tgt = faces_tgt[0]
        print("[INFO] Detected target face[0]:", face_tgt)

        bbox_tgt = face_tgt.get('bbox')
        if bbox_tgt is None or len(bbox_tgt) != 4:
            return {"status": "error", "message": "Invalid or missing bounding box for target face."}

        x1, y1, x2, y2 = map(int, bbox_tgt)
        h_tgt, w_tgt = tgt_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_tgt, x2), min(h_tgt, y2)

        if x2 <= x1 or y2 <= y1:
            return {"status": "error", "message": "Bounding box resulted in invalid region."}

        cropped_np = np.array(src_img)[y1_src:y2_src, x1_src:x2_src]

        if cropped_np.size == 0:
            return {"status": "error", "message": "Cropped face is empty."}

        face_img = Image.fromarray(cropped_np)

        prompt = request.prompt.strip() if request.prompt else "best quality, high quality"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        print("[INFO] Using prompt:", prompt)
        print("[INFO] Using negative_prompt:", negative_prompt)

        # Generate a new face only (square output)
        # Make width and height divisible by 8
        w, h = face_img.width, face_img.height
        w -= w % 8
        h -= h % 8

        new_face_img = ip_adapter.generate(
            pil_image=face_img,
            face_image=face_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_samples=1,
            num_inference_steps=40,
            seed=42,
            height=h,
            width=w
        )[0]

        new_face_np = np.array(new_face_img.resize((x2 - x1, y2 - y1)))

        # Blend the generated face into the original target image with a soft mask
        mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        cv2.circle(mask, ((x2 - x1) // 2, (y2 - y1) // 2), min((x2 - x1), (y2 - y1)) // 2, (255, 255, 255), -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0

        face_region = tgt_np[y1:y2, x1:x2].astype(np.float32)
        blended_face = (mask * new_face_np + (1 - mask) * face_region).astype(np.uint8)
        tgt_np[y1:y2, x1:x2] = blended_face
        final_image = Image.fromarray(tgt_np).convert("RGB")

        from fastapi.responses import StreamingResponse

        output_buffer = BytesIO()
        final_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")

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
