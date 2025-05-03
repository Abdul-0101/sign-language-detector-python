# app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from inference_classifier import classify_frame

app = FastAPI()

# CORS middleware for local frontend (change if deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    frame_data = data.get("frame")
    if not frame_data:
        return {"error": "No frame received"}

    # Remove header and decode base64
    try:
        encoded = frame_data.split(",")[1]
        decoded = base64.b64decode(encoded)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": f"Invalid frame data: {str(e)}"}

    # Get prediction from inference logic
    result = classify_frame(frame)
    return result
