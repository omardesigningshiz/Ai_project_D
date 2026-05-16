# This code is used to collect data for training a machine learning model.
# It uses the OpenCV library to capture images from the webcam and saves them in a specified directory.
# The user is prompted to press 's' to start collecting data for each class, and the images are saved in subdirectories corresponding to each class.
import cv2
import os

DATA_DIR = "data"

# Create the main data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 30  # Number of letters in the arabic language
data_size = 200  # Number of images per class
cap = cv2.VideoCapture(0)
for i in range(number_of_classes):

    # Create a subdirectory for each class
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))

    print(f"Collecting data for class {i}...")
    done = False
    while True:                                    # Wait for the user to press 's' to start collecting data
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "s" to start collecting data.',
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break
    count = 0
    while count < data_size:            # Collect data until the specified number of images is reached
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), f"{count}.jpg"), frame)
        count += 1
cap.release()
cv2.destroyAllWindows()
