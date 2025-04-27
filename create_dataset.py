import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Use only the right-hand images (folders named "A", "B", ..., "Z" under data/right)
DATA_DIR = './data/right'

data = []
labels = []

# Loop over each folder in ./data/right (each folder is a letter)
for dir_ in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder {dir_} ...")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read {img_path}")
            continue

        # Convert to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # For each detected hand, compute a simple feature vector:
            # For each landmark, subtract the minimum x and y to normalize
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                feature_vector = []
                for lm in hand_landmarks.landmark:
                    feature_vector.append(lm.x - min(x_coords))
                    feature_vector.append(lm.y - min(y_coords))
                data.append(feature_vector)
                labels.append(dir_)  # label is the folder name (e.g. "A")
                # (Assumes one hand per image; if multiple hands occur, only the first is used)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset saved as data.pickle")
