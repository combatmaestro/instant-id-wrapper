FastAPI InstantID Face Swap API

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from io import BytesIO
from PIL import Image
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_instantid_pipeline():
    print("[INFO] Simulating InstantID model loading...")
    return lambda src, tgt, prompt: tgt  # dummy: returns target as result

model_pipeline = load_instantid_pipeline()

@app.get("/")
def root():
    return {"message": "InstantID API is running"}

@app.post("/swap-face")
async def swap_face(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    try:
        src_bytes = await source_image.read()
        tgt_bytes = await target_image.read()
        src_image = Image.open(BytesIO(src_bytes)).convert("RGB")
        tgt_image = Image.open(BytesIO(tgt_bytes)).convert("RGB")

        result_image = model_pipeline(src_image, tgt_image, prompt)
        output_buffer = BytesIO()
        result_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return {
            "status": "success",
            "message": "Face swapped (mock). Replace with actual InstantID logic."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}        

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
