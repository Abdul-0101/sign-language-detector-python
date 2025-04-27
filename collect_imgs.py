import os
import cv2

DATA_DIR = './data'
DATASET_SIZE = 500  # Number of images per class


def setup_directories():
    for hand in ['left', 'right']:
        hand_path = os.path.join(DATA_DIR, hand)
        os.makedirs(hand_path, exist_ok=True)
        for i in range(26):  # Create folders for A-Z
            class_dir = os.path.join(hand_path, chr(65 + i))
            os.makedirs(class_dir, exist_ok=True)


def capture_images_for_sign(letter, hand, cap):
    class_dir = os.path.join(DATA_DIR, hand, letter.upper())
    print(f"\nCollecting {DATASET_SIZE} images for {letter.upper()} ({hand.capitalize()})...")
    count = 0
    while count < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Capturing {letter.upper()} [{count + 1}/{DATASET_SIZE}]", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        cv2.imwrite(os.path.join(class_dir, f"{count}.jpg"), frame)
        count += 1
        cv2.waitKey(25)
    print(f"Image collection complete for {letter.upper()} ({hand.capitalize()}).")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")

    setup_directories()

    while True:
        # State variables for user input
        letter = ""
        hand = ""

        # Input loop: wait for letter and hand input while showing the camera frame
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # Display instructions and current input on the frame
            cv2.putText(frame, f"Enter Letter (A-Z): {letter}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if letter:
                cv2.putText(frame, f"Enter Hand (L/R): {hand}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press ENTER to confirm, BACKSPACE to reset", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit if 'q' is pressed
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            # If ENTER is pressed and both letter and hand are set, exit input loop
            if key == 13 and letter and hand:
                break

            # If BACKSPACE is pressed, reset input
            if key == 8:
                letter = ""
                hand = ""
                continue

            # If letter is not yet set and key is a letter (A-Z or a-z), capture it
            if not letter and ((65 <= key <= 90) or (97 <= key <= 122)):
                letter = chr(key).upper()
                continue

            # If letter is set but hand is not, check for hand input (L/l or R/r)
            if letter and not hand and key in [ord('L'), ord('l'), ord('R'), ord('r')]:
                hand = "left" if chr(key).upper() == 'L' else "right"
                continue

        # Start capturing images for the current sign
        capture_images_for_sign(letter, hand, cap)

        # After capturing, ask user if they want to continue with another sign or quit
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press 'c' to capture another sign or 'q' to quit", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                break


if __name__ == "__main__":
    main()
