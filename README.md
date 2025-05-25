# InstantID API Wrapper (FastAPI)

A minimal API wrapper to integrate InstantID face blending via `/swap-face`.

## How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Folder Structure

```
instantid-api-wrapper/
├── main.py
├── requirements.txt
├── README.md
└── models/
    ├── sdxl/
    ├── controlnet/
    └── ip-adapter/
```

> Place your model files into the `models/` subfolders.
