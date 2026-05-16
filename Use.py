import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display
from Classifier import HandSignClassifier


# 2. Setup Device and Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandSignClassifier().to(device)
model.load_state_dict(torch.load(
    'hand_sign_classifier.pth', map_location=device))
model.eval()  # Set model to evaluation mode (turns off dropout)

# 3. Label Dictionary (Based on the ArSL dataset mapping)
labels_dict = {
    0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح',
    6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر', 10: 'ز', 11: 'س',
    12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ',
    17: 'ع', 18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل', 23: 'م', 24: 'ن', 25: 'ه', 26: 'و', 27: 'ي',
    28: 'ة', 29: 'لا'
}

# 4. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Notice: static_image_mode is False because we are using a live video feed
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 5. Start Webcam
cap = cv2.VideoCapture(0)

print("Starting video stream... Press 'q' to quit.")


def put_arabic_text(img, text, position, font_size=50, color=(0, 255, 0)):
    # 1. Reshape the text to join characters properly (important for 'لا')
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # 2. Convert OpenCV image (NumPy array) to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 3. Load a Windows font that supports Arabic (Arial is standard)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Warning: Arial font not found, text might not render correctly.")
        font = ImageFont.load_default()

    # 4. Draw the text
    draw.text(position, bidi_text, font=font, fill=color)

    # 5. Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


print("Starting video stream...")
print("-> Press 'q' to quit.")
print("-> Press 'SPACEBAR' to add a space.")
print("-> Press 'c' to clear the current text.")

# --- NEW VARIABLES FOR WORD BUILDING ---
current_word = ""
last_predicted_char = ""
frames_held = 0
FRAMES_REQUIRED = 15  # Hold a sign for 15 frames (~0.5 sec) to register it


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(frame_rgb)

    current_frame_prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the camera feed
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Data Preprocessing
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_), min(y_)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            # Predict!
            if len(data_aux) == 42:
                input_tensor = torch.tensor(
                    [data_aux], dtype=torch.float32).to(device)

                with torch.no_grad():
                    prediction = model(input_tensor)

                predicted_class = torch.argmax(prediction, dim=1).item()
                predicted_character = labels_dict.get(
                    predicted_class, "Unknown")
                current_frame_prediction = predicted_character

                # Get bounding box coordinates for the floating text
                x1 = int(min_x * W)
                y1 = int(min_y * H) - 20

                # Draw the LIVE prediction hovering over the hand
                frame = put_arabic_text(
                    frame, predicted_character, (x1, y1), font_size=50, color=(0, 255, 0))

    # --- THE WORD BUILDING LOGIC ---
    if current_frame_prediction != "":
        if current_frame_prediction == last_predicted_char:
            frames_held += 1
            # If held long enough, add it to the string ONCE
            if frames_held == FRAMES_REQUIRED:
                current_word += current_frame_prediction
                print(f"Current Text: {current_word}")
        else:
            # If the sign changes, reset the counter
            last_predicted_char = current_frame_prediction
            frames_held = 0
    else:
        # If no hand is detected, reset the counter
        last_predicted_char = ""
        frames_held = 0

    # Draw the built sentence at the bottom of the screen (Blue text)
    if current_word:
        frame = put_arabic_text(frame, current_word,
                                (20, H - 80), font_size=60, color=(255, 0, 0))

    # Show the final image
    cv2.imshow('Arabic Sign Language Translator', frame)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Spacebar for a space
        current_word += " "
        print(f"Current Text: {current_word}")
    elif key == ord('c'):  # 'c' to clear text
        current_word = ""
        print("Text cleared.")

cap.release()
cv2.destroyAllWindows()
