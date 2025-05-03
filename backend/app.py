
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
import pickle
import os
from inference_classifier import predict_word, model, hands, get_bounding_box

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load dictionary (from your code)
from inference_classifier import dictionary

@app.post("/infer/")
async def infer_image(file: UploadFile = File(...)):
    # read bytes → cv2 image
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return {"letter": "", "word": ""}

    hand = results.multi_hand_landmarks[0]
    # bounding‐box filtering (as in your script)
    xs = [lm.x for lm in hand.landmark]; ys = [lm.y for lm in hand.landmark]
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    area = (max_x - min_x)*(max_y - min_y)
    if not (0.02 <= area <= 0.2):
        return {"letter": "", "word": ""}

    # build feature vector
    data_aux, x_coords = [], xs
    for lm in hand.landmark:
        data_aux += [lm.x - min(x_coords), lm.y - min(ys)]
    letter = model.predict([np.asarray(data_aux)])[0]
    word = predict_word(letter)
    return {"letter": letter, "word": word}
