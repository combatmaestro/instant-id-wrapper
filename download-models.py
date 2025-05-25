

from huggingface_hub import snapshot_download
import os

os.makedirs("./models/sdxl", exist_ok=True)
os.makedirs("./models/ip-adapter", exist_ok=True)
os.makedirs("./models/controlnet", exist_ok=True)

print("📦 Downloading SDXL...")
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="./models/sdxl",
    allow_patterns=[
        "model_index.json",
        "scheduler_config.json",
        "config.json",
        "**/*.safetensors",
        "**/tokenizer_config.json",
        "**/vocab.json",
        "**/merges.txt",
        "**/special_tokens_map.json"
    ],
    local_dir_use_symlinks=False
)

print("📦 Downloading IP-Adapter (SDXL)...")
snapshot_download(
    repo_id="h94/IP-Adapter",
    local_dir="./models/ip-adapter",
    local_dir_use_symlinks=False,
    allow_patterns=["models/ip-adapter_sdxl.bin"]
)

print("📦 Downloading ControlNet for InstantID...")
snapshot_download(
    repo_id="InstantX/InstantID",
    local_dir="./models/controlnet",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "controlnet/control_v11p_sd15_openpose.pth",
        "ip-adapter.bin",
        "insightface_model/*"
    ]
)

print("✅ All models downloaded successfully.")
