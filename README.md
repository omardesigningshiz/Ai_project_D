# Arabic Sign Language (ArSL) Live Translator

This project is an end-to-end machine learning pipeline that translates Arabic sign language from a live webcam feed into text. It uses MediaPipe for hand landmark extraction and a custom Feedforward Neural Network built with PyTorch for real-time classification.

## ⚙️ How It Works (Project Structure)

The project is broken down into four main scripts that handle the entire workflow:

1. **`Data collection.py`**: Captures images from a live webcam and organizes them into folders (0-29) representing the Arabic alphabet. This allows for custom dataset creation tailored to specific lighting or environments.
   
2. **`Data_set.py`**: Processes both the custom webcam images and a supplementary Kaggle ArSL dataset. It uses MediaPipe to extract 21 (x, y) hand landmarks, normalizes the coordinates for translation invariance, and saves the combined feature set into a `data.pkl` file.
   
3. **`Classifier.py`**: Trains a 3-layer Feedforward Neural Network using PyTorch to classify the 42 extracted coordinate features into one of the 30 Arabic letters. It utilizes Dropout for regularization and generates convergence plots and a confusion matrix to evaluate performance.
   
4. **`Use.py`**: The live inference script. It captures webcam video, extracts landmarks in real-time, and displays the translated letters using PyTorch model weights. 
   * **Features:** Utilizes a frame-buffer logic (15 frames) to transform isolated letters into continuous, stable text. 
   * **Controls:** Press `SPACEBAR` to add a space, `c` to clear the current text, and `q` to quit the application.

## 📊 Results and Evaluation

* **Overall Accuracy:** 97%
* **Architecture:** 42 Input -> 128 Hidden -> 64 Hidden -> 30 Output
* **Regularization:** 20% Dropout

---

## 🛠️ Installation and Setup

To run this project locally, you will need Python installed. It is highly recommended to use a virtual environment.

**1. Clone the repository:**
```
git clone [https://github.com/omardesigningshiz/Ai_project_D.git](https://github.com/omardesigningshiz/Ai_project_D.git)
cd Ai_project_D
```
2. Install the required dependencies:
Run the following command to install the necessary machine learning, computer vision, and Arabic text formatting libraries:
```
pip install torch torchvision torchaudio opencv-python mediapipe numpy pillow scikit-learn matplotlib seaborn kagglehub arabic-reshaper python-bidi
```
## 🚀 How To Run

To train the model from scratch and run the live translator, execute the scripts in the following order:

Step 1: Collect Custom Data (Optional but recommended)


python "Data collection.py"
Follow the on-screen prompts to press 's' and hold signs for the camera.

Step 2: Process the Dataset


python Data_set.py
This will download the Kaggle dataset, combine it with your local data, run MediaPipe to extract landmarks, and generate data.pkl.

Step 3: Train the Neural Network


python Classifier.py
This trains the PyTorch model, saves the weights as hand_sign_classifier.pth, and generates your performance evaluation graphs.

Step 4: Run the Live Translator

python Use.py
This opens your webcam and begins translating your hand signs into Arabic text in real-time.
***

