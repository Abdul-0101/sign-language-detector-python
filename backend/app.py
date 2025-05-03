from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, io
from PIL import Image
from inference_classifier import classify_frame

app = FastAPI()

# ─── CORS ──────────────────────────────────────────────────────────────────────
# allow your frontend origin (and localhost for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sign-language-web-03n4.onrender.com",  # your deployed React
        "http://localhost:3000",                        # for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ────────────────────────────────────────────────────────────────────────────────

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = classify_frame(frame)
    return result
