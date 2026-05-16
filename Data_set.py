import os
import cv2
import mediapipe as mp
import pickle
import kagglehub as kh
import warnings

# Ignore that specific Protobuf warning to clean up your terminal
warnings.filterwarnings("ignore", category=UserWarning,
                        module='google.protobuf.symbol_database')

Local_DATA_DIR = "data"
# Download Kaggle data
print("Checking Kaggle dataset...")
kaggle_raw = kh.dataset_download(
    "birafaneimane/arabic-sign-language-alphabet-arsl-dataset")

# AUTOMATIC SEARCH: Find the folder that actually contains the numbered subfolders
kaggle_path = None
for root, dirs, files in os.walk(kaggle_raw):
    # This dataset uses folders 0-29. Let's look for folder '0'
    if '0' in dirs and '15' in dirs:
        kaggle_path = root
        break

SOURCES = [Local_DATA_DIR, kaggle_path]
print(f"Final Sources to process: {SOURCES}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

for base_path in SOURCES:
    if not base_path or not os.path.exists(base_path):
        print(f"Skipping: {base_path} (Path does not exist)")
        continue

    print(f"--- Entering Source: {base_path} ---")

    folders = [f for f in os.listdir(
        base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"Found {len(folders)} subfolders in {base_path}")

    for dir_ in folders:
        subfolder_path = os.path.join(base_path, dir_)

        # Count images in this folder
        images = os.listdir(subfolder_path)
        print(f"  Processing Class {dir_}: {len(images)} images found.")

        for img_name in images:
            full_img_path = os.path.join(subfolder_path, img_name)
            img = cv2.imread(full_img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(int(dir_))

# Save
if len(data) > 0:
    with open("data.pkl", "wb") as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"\nSUCCESS! Total samples saved to data.pkl: {len(data)}")
else:
    print("\nFAILED: No hands were detected in any images. Check image quality.")
