import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time
import os
import tkinter as tk
from tkinter import simpledialog
import pyttsx3
import speech_recognition as sr

# -------------------------------
# Text-To-Speech (TTS) Setup
# -------------------------------
engine = pyttsx3.init()
# Slow down the speech rate (default is around 200 wpm)
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
current_voice_index = 0  # For cycling voices


def speak(text):
    engine.say(text)
    engine.runAndWait()


def cycle_voice():
    global current_voice_index
    if voices:
        current_voice_index = (current_voice_index + 1) % len(voices)
        engine.setProperty('voice', voices[current_voice_index].id)
        voice_name = voices[current_voice_index].name
        confirmation = f"Voice changed to {voice_name}"
        print(confirmation)
        speak(confirmation)
        return voice_name
    return "Default"


# -------------------------------
# Voice Integration Setup (Requires PyAudio)
# -------------------------------
voice_enabled = True
try:
    import speech_recognition as sr
except ModuleNotFoundError:
    print(
        "WARNING: PyAudio (required for voice commands) is not installed. Please install it using pipwin or a compatible wheel.")
    voice_enabled = False

voice_command = None
voice_lock = threading.Lock()


def voice_listener():
    """Continuously listens for voice commands and updates a global variable."""
    global voice_command
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio)
                with voice_lock:
                    voice_command = command.strip()
        except Exception:
            with voice_lock:
                voice_command = None
        time.sleep(0.1)


if voice_enabled:
    voice_thread = threading.Thread(target=voice_listener, daemon=True)
    voice_thread.start()

# -------------------------------
# Dictionary with Persistent Storage
# -------------------------------
DICTIONARY_FILE = "dictionary.txt"


def load_dictionary():
    if os.path.exists(DICTIONARY_FILE):
        with open(DICTIONARY_FILE, "r") as file:
            return set(word.strip().upper() for word in file.readlines())
    return set()


def save_dictionary(dictionary):
    with open(DICTIONARY_FILE, "w") as file:
        for word in sorted(dictionary):
            file.write(f"{word}\n")


dictionary = load_dictionary()
if not dictionary:
    dictionary.update({
        "HELLO", "WORLD", "HOW", "ARE", "YOU", "HI", "HELP", "HOUSE", "PYTHON",
        "PROGRAMMING", "EXAMPLE", "TEST", "GOOD", "MORNING", "EVERYONE",
        "ABDUL", "AHMED", "MOHAMMED", "HASSAN", "OMAR", "KUMAR", "RAJ", "SURESH",
        "RAVI", "ANIL", "SACHIN", "PRAVEEN", "SUNITA", "PRIYA", "NEHA", "POOJA",
        "KAVITA", "SANDHYA", "DEEPA", "MANISH", "VINEET", "ROHIT", "AMOL", "SAMEER",
        "NAVEEN", "GANESH", "RAJESH", "RESHMA", "PALLAVI", "MEGHNA",
        "JAMES", "JOHN", "MARY", "PATRICK", "DAVID", "SARAH", "ELIZABETH",
        "ROBERT", "LINDA", "MICHAEL", "JOSEPH", "GABRIEL", "AYESHA", "FATIMA",
        "KHALID", "SAMIR", "ZAINAB", "JESUS", "ISRAEL", "RACHEL", "BODHI", "SIDDHARTHA"
    })
    save_dictionary(dictionary)


def predict_word(prefix):
    if not prefix:
        return ""
    prefix = prefix.upper()
    matches = [word for word in dictionary if word.startswith(prefix)]
    if matches:
        return max(matches, key=len)
    return ""


def add_word_to_dictionary(word):
    word = word.upper()
    if word not in dictionary:
        dictionary.add(word)
        save_dictionary(dictionary)


# -------------------------------
# Tkinter Setup for Correction Input
# -------------------------------
root = tk.Tk()
root.withdraw()


def get_correction_input():
    return simpledialog.askstring("Correction", "Enter the correct word:")


# -------------------------------
# Load the Trained Model (Random Forest in this example)
# -------------------------------
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Increase min_detection_confidence to reduce false detections (e.g., from your dress)
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

cap = cv2.VideoCapture(0)


# -------------------------------
# Helper: Compute Bounding Box of Hand Landmarks
# -------------------------------
def get_bounding_box(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


# -------------------------------
# Text Buffers and Control Variables
# -------------------------------
final_text = ""  # Accumulated paragraph
current_text = ""  # Current word being formed
predicted_word = ""
predicted_letter = ""

stable_detection_threshold = 3  # Frames required to confirm a letter
stable_count = 0
last_detected_letter = ""
waiting_for_hand_removal = False
hand_absent_count = 0
hand_absent_threshold = 2  # Consecutive frames with no hand required to reset


def update_model_with_feedback(correction):
    print(f"Updating model with correction: {correction}")
    # Placeholder for reinforcement/active learning logic


# -------------------------------
# Create Output Window for Text Info
# -------------------------------
output_height, output_width = 600, 800  # Increased height for full instructions
output_img = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

instructions = (
    "Key Commands:\n"
    "s: Accept predicted word\n"
    "Space: Confirm current word & insert space (TTS speaks word)\n"
    "Backspace: Delete last character\n"
    "Enter: Confirm current word & add newline (TTS speaks word)\n"
    "c: Correction (opens input dialog)\n"
    "v: Speak final text\n"
    "f: Cycle TTS voice\n"
    "q: Quit"
)

# -------------------------------
# Main Loop
# -------------------------------
while True:
    detected_letter = ""  # Always initialize for each frame
    data_aux = []
    x_coords = []
    y_coords = []

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_present = results.multi_hand_landmarks is not None

    # Waiting for hand removal logic
    if waiting_for_hand_removal:
        if not hand_present:
            hand_absent_count += 1
            if hand_absent_count >= hand_absent_threshold:
                waiting_for_hand_removal = False
                hand_absent_count = 0
                last_detected_letter = ""
        else:
            hand_absent_count = 0

    # Process voice commands if enabled
    if voice_enabled:
        with voice_lock:
            current_voice = voice_command
            voice_command = None
        if current_voice:
            cmd = current_voice.upper()
            if cmd.startswith("CORRECTION:"):
                correct_word = cmd.split("CORRECTION:")[1].strip()
                if correct_word:
                    add_word_to_dictionary(correct_word)
                    update_model_with_feedback(correct_word)
                    predicted_word = correct_word
                    print(f"Voice Correction Applied: {predicted_word}")
            elif cmd == "CONFIRM":
                if current_text:
                    final_text += current_text + " "
                    speak(current_text)
                    current_text = ""
            elif cmd == "DELETE":
                if current_text:
                    current_text = current_text[:-1]
                elif final_text:
                    final_text = final_text[:-1]
            elif cmd == "ACCEPT":
                if predicted_word:
                    current_text = predicted_word
            elif cmd == "QUIT":
                break

    # Letter detection via hand landmarks with additional bounding box filtering
    if hand_present and not waiting_for_hand_removal:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Get bounding box to filter out detections (e.g., dress) by checking area
        min_x, min_y, max_x, max_y = get_bounding_box(hand_landmarks.landmark)
        bbox_area = (max_x - min_x) * (max_y - min_y)
        # Only proceed if bounding box area is within plausible range (adjust these thresholds as needed)
        if 0.02 <= bbox_area <= 0.2:
            for lm in hand_landmarks.landmark:
                x_coords.append(lm.x)
                y_coords.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_coords))
                data_aux.append(lm.y - min(y_coords))
            if data_aux:
                detected_letter = model.predict([np.asarray(data_aux)])[0]
        else:
            detected_letter = ""
        if detected_letter:
            if detected_letter == last_detected_letter:
                stable_count += 1
            else:
                last_detected_letter = detected_letter
                stable_count = 1
            if stable_count >= stable_detection_threshold:
                current_text += detected_letter
                waiting_for_hand_removal = True
                stable_count = 0
    else:
        if not waiting_for_hand_removal:
            stable_count = 0
            last_detected_letter = ""

    predicted_word = predict_word(current_text)

    if current_text and current_text not in dictionary:
        add_word_to_dictionary(current_text)

    # Build output window content
    output_img[:] = 255  # Clear to white
    y0, dy = 30, 30
    lines = []
    if final_text:
        lines.append("Paragraph:")
        for line in final_text.split("\n"):
            lines.append(line)
    lines.append("Current Word: " + current_text)
    lines.append("Predicted: " + predicted_word)
    lines.append("Status: " + ("Hand Present" if hand_present else "Hand Not Present"))
    lines.append("Current TTS Voice: " + voices[current_voice_index].name)
    for i, line in enumerate(lines):
        cv2.putText(output_img, line, (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    y_instruction = y0 + len(lines) * dy + 20
    for instr_line in instructions.split("\n"):
        cv2.putText(output_img, instr_line, (10, y_instruction),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        y_instruction += 20

    cv2.putText(frame, f"Detected: {detected_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if waiting_for_hand_removal:
        cv2.putText(frame, "Remove hand to detect next letter", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Sign Language Detector", frame)
    cv2.imshow("Output", output_img)

    # Key-based controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if predicted_word:
            current_text = predicted_word
    elif key == 32:  # Spacebar: confirm current word & insert space
        if current_text:
            final_text += current_text + " "
            speak(current_text)
            current_text = ""
        else:
            if final_text and not final_text.endswith(" "):
                final_text += " "
    elif key == 8:  # Backspace: delete last character
        if current_text:
            current_text = current_text[:-1]
        elif final_text:
            final_text = final_text[:-1]
    elif key == 13:  # Enter: confirm current word & add newline
        if current_text:
            final_text += current_text + "\n"
            speak(current_text)
            current_text = ""
        else:
            final_text += "\n"
    elif key == ord('c'):
        correction = get_correction_input()
        if correction and predicted_word:
            add_word_to_dictionary(correction)
            update_model_with_feedback(correction)
            predicted_word = correction
            print(f"Correction Applied: {predicted_word}")
    elif key == ord('v'):
        if final_text:
            speak(final_text)
        elif current_text:
            speak(current_text)
        else:
            speak("No text to speak.")
    elif key == ord('f'):
        new_voice = cycle_voice()
        print(f"Switched to voice: {new_voice}")

cap.release()
cv2.destroyAllWindows()
