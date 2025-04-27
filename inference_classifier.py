import cv2
import numpy as np
import pickle
import os
from gtts import gTTS
#import speech_recognition as sr

# ─── CONFIGURATION ─────────────────────
MODEL_PATH = 'model.p'
TTS_OUTPUT = os.path.join("static", "tts.mp3")

# ─── LOAD THE MODEL ─────────────────────
model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']

# ─── GLOBAL VARIABLES ───────────────────
current_word = ''

# ─── TEXT-TO-SPEECH FUNCTION ────────────
def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(TTS_OUTPUT)
    except Exception as e:
        print(f"[TTS ERROR] {e}")

# ─── WORD HANDLING FUNCTIONS ────────────
def update_current_word(letter):
    global current_word
    if letter == 'space':
        current_word += ' '
    elif letter == 'del':
        current_word = current_word[:-1]
    else:
        current_word += letter.lower()

def get_current_word():
    global current_word
    return current_word

def reset_current_word():
    global current_word
    current_word = ''

# ─── INFERENCE / PREDICTION FUNCTION ─────
def predict(frame):
    # 1. Flip the frame correctly (left-right)
    frame = cv2.flip(frame, 1)

    # 2. Define ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # 3. Preprocess the ROI
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))  # model expects 64x64
    roi = roi.reshape(1, 64, 64, 1)
    roi = roi / 255.0  # Normalize

    # 4. Predict
    prediction = model.predict(roi)
    pred_class = np.argmax(prediction)
    classes = model_dict['classes']  # Assuming your model dict has classes

    letter = classes[pred_class]
    confidence = prediction[0][pred_class]

    return letter, confidence, frame

# ─── SPEECH RECOGNITION FUNCTION (OPTIONAL) ─────
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        recognized_text = recognizer.recognize_google(audio)
        print("You said:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition error; {e}")
        return ""
