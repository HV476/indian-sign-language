# 🤟 Indian Sign Language → Text Translator

> A real-time Indian Sign Language (ISL) recognition system that uses computer vision and deep learning to recognize hand gestures and translate them into text.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)](https://ai.google.dev/edge/mediapipe/solutions/guide)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)](https://keras.io/)

---

## 📌 Overview

Communication barriers can make everyday interactions difficult for people who use sign language.

This project explores an AI-powered approach to **Indian Sign Language recognition**, using a webcam to capture hand gestures, extracting hand landmarks with **MediaPipe**, and classifying sequences of gestures using a **Long Short-Term Memory (LSTM)** neural network.

The system processes a sequence of hand movements and predicts the corresponding alphabet character in real time.

### 🎯 Goal

Build a computer-vision-based system capable of:

* 📷 Capturing hand gestures through a webcam
* ✋ Detecting hand landmarks using MediaPipe
* 🧮 Converting landmarks into numerical feature vectors
* 🧠 Learning temporal gesture patterns using LSTM
* 🔤 Recognizing Indian Sign Language alphabet gestures
* ⚡ Performing real-time predictions
* 📝 Displaying the predicted character and confidence

---

## ✨ Features

* **Real-time gesture recognition**
* **Webcam-based input**
* **MediaPipe hand landmark detection**
* **63-dimensional hand landmark representation**
* **LSTM-based sequence classification**
* **26 alphabet classes — A to Z**
* **Pre-trained model included**
* **Confidence threshold for predictions**
* **Custom dataset collection pipeline**
* **Training pipeline for creating a new model**

---

## 🧠 How It Works

The system follows a simple computer vision → deep learning pipeline:

```text
                Webcam / Images
                       │
                       ▼
              ┌─────────────────┐
              │     OpenCV      │
              │ Frame Capture   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    MediaPipe    │
              │ Hand Detection  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Landmark        │
              │ Extraction      │
              │ 21 × (x,y,z)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ 30-Frame        │
              │ Sequence        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   LSTM Model    │
              │ 64 → 128 → 64   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Softmax Output  │
              │ A → Z           │
              └────────┬────────┘
                       │
                       ▼
                 Predicted Text
```

The implementation extracts 21 hand landmarks with `(x, y, z)` coordinates, resulting in **63 numerical features per frame**. The recognition model then processes sequences of 30 frames.

---

## 🏗️ Model Architecture

The project uses a stacked LSTM architecture:

```text
Input
30 frames × 63 features
        │
        ▼
LSTM
64 units
        │
        ▼
LSTM
128 units
        │
        ▼
LSTM
64 units
        │
        ▼
Dense
64 units
        │
        ▼
Dense
32 units
        │
        ▼
Dense
26 classes
        │
        ▼
Softmax
A - Z
```

The training script uses:

* **3 LSTM layers**
* Dense layers with **64 and 32 neurons**
* Final **26-class softmax output**
* **Adam optimizer**
* **Categorical cross-entropy loss**
* **Categorical accuracy**
* TensorBoard logging

The repository's training configuration uses 1,000 epochs.

---

## 🔤 Supported Gestures

The current model is configured for the English alphabet:

```text
A  B  C  D  E  F  G
H  I  J  K  L  M  N
O  P  Q  R  S  T  U
V  W  X  Y  Z
```

The action labels are explicitly defined as the 26 alphabet classes in the source code.

---

## 📁 Project Structure

```text
indian-sign-language/
│
├── indian sign language/
│   └── Indian-Sign-language-to-text-translater/
│       │
│       ├── Dataset/
│       │
│       ├── Source code/
│       │   ├── app.py
│       │   ├── collectdata.py
│       │   ├── data.py
│       │   ├── function.py
│       │   ├── trainmodel.py
│       │   ├── model.h5
│       │   ├── model.json
│       │   └── readme
│       │
│       └── README.md
│
└── README.md
```

The source-code directory contains separate components for data collection, preprocessing, model training, reusable computer-vision functions, and real-time inference.

---

## ⚙️ Tech Stack

| Technology             | Purpose                                         |
| ---------------------- | ----------------------------------------------- |
| **Python**             | Core programming language                       |
| **OpenCV**             | Image and webcam processing                     |
| **MediaPipe**          | Hand landmark detection                         |
| **NumPy**              | Numerical processing and feature representation |
| **TensorFlow / Keras** | Deep learning model                             |
| **LSTM**               | Temporal gesture classification                 |
| **TensorBoard**        | Training visualization                          |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HV476/indian-sign-language.git
```

```bash
cd indian-sign-language
```

### 2. Navigate to the project

```bash
cd "indian sign language/Indian-Sign-language-to-text-translater/Source code"
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

Install the required Python packages:

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install tensorflow
pip install keras
```

> For reproducible development, pin compatible package versions in a `requirements.txt` file.

---

# 📷 Running the Translator

Once the dependencies and model files are available, run:

```bash
python app.py
```

The application opens the webcam and continuously processes frames.

The inference pipeline:

1. Captures a webcam frame.
2. Crops the active region.
3. Detects the hand using MediaPipe.
4. Extracts hand landmarks.
5. Builds a rolling sequence of frames.
6. Sends the sequence to the LSTM model.
7. Calculates the predicted class.
8. Applies a confidence threshold.
9. Displays the predicted character.

The current inference implementation maintains a rolling sequence of **30 frames** and uses a confidence threshold of **0.8** before accepting a prediction.

### Quit the application

Press:

```text
Q
```

to exit the webcam window.

---

# 🧪 Training Your Own Model

The repository also contains the scripts required to build the dataset and train the recognition model.

### Step 1 — Collect data

Use:

```bash
python collectdata.py
```

The data collection script captures images from the webcam and stores gesture samples for alphabet classes.

### Step 2 — Generate landmark data

The preprocessing pipeline uses MediaPipe to convert images into hand landmark coordinates.

The resulting feature vectors are stored as NumPy arrays under the project's `MP_Data` directory.

### Step 3 — Train the model

Run:

```bash
python trainmodel.py
```

The training pipeline loads the generated landmark sequences, creates training/testing splits, trains the LSTM model, and saves the resulting model architecture and weights.

The generated files are:

```text
model.json
model.h5
```

---

# 📊 Data Representation

Each detected hand contains **21 landmarks**.

Every landmark contains:

```text
x
y
z
```

Therefore:

```text
21 landmarks × 3 coordinates
= 63 features
```

A temporal sample consists of:

```text
30 frames × 63 features
```

which is supplied to the LSTM network for classification.

---

# 🧩 Core Components

### `function.py`

Contains the core computer-vision utilities:

* MediaPipe initialization
* Image color conversion
* Hand detection
* Landmark drawing
* Landmark extraction
* Dataset configuration
* Gesture labels

The project uses MediaPipe Hands with configurable detection and tracking confidence thresholds.

### `collectdata.py`

Responsible for collecting gesture images using a webcam and organizing samples by alphabet class.

### `data.py`

Converts gesture images into MediaPipe landmark sequences and saves the extracted numerical features as `.npy` files.

### `trainmodel.py`

Builds and trains the stacked LSTM classifier and exports:

```text
model.json
model.h5
```

### `app.py`

Runs real-time inference using the webcam, MediaPipe and the trained model.

---

# 🔬 Machine Learning Pipeline

```text
Raw Image
   │
   ▼
OpenCV
   │
   ▼
MediaPipe Hands
   │
   ▼
21 Hand Landmarks
   │
   ▼
63-Dimensional Feature Vector
   │
   ▼
30 Frame Sequence
   │
   ▼
LSTM Network
   │
   ▼
Softmax Classification
   │
   ▼
A-Z Prediction
```

This approach avoids feeding raw images directly into the classifier. Instead, the system uses MediaPipe's structured hand landmarks as a compact representation of hand pose and feeds temporal landmark sequences into the LSTM.

---

# 💡 Why LSTM?

Sign-language gestures can contain temporal information. A single frame may not always provide enough information to distinguish between movements.

LSTMs are designed for sequential data and can learn patterns across multiple frames.

In this project:

```text
Frame 1
   ↓
Frame 2
   ↓
Frame 3
   ↓
 ...
   ↓
Frame 30
   ↓
LSTM
   ↓
Gesture Classification
```

This makes the model suitable for sequence-based gesture recognition.

---

# 🛠️ Possible Improvements

The current implementation provides a foundation for ISL recognition, but there are several directions in which it could be extended.

### Recognition

* Support complete ISL vocabulary
* Recognize words and phrases instead of individual alphabet characters
* Improve multi-hand gesture recognition
* Add dynamic gestures
* Add gesture segmentation
* Improve robustness under different lighting conditions

### Machine Learning

* Add data augmentation
* Use a validation set during training
* Add early stopping
* Tune the LSTM architecture
* Compare LSTM with GRU/Transformer-based models
* Track precision, recall and F1-score
* Build a confusion matrix

### Application

* Build a browser-based interface
* Add text-to-speech
* Add speech-to-sign translation
* Add Hindi and other Indian language support
* Expose the model through a REST API
* Deploy the inference service using Docker
* Build a mobile application

### Production

```text
Webcam
   ↓
Inference API
   ↓
ISL Recognition Model
   ↓
Prediction Service
   ↓
Text / Speech / Translation
```

---

# 🌍 Real-World Applications

This technology can potentially be used as a foundation for:

* 🏫 Educational accessibility tools
* 🏥 Healthcare communication
* 🏢 Workplace accessibility
* 🤝 Public-service communication
* 📱 Mobile accessibility applications
* 🧑‍🏫 ISL learning platforms
* 💬 Real-time communication assistants

---

# ⚠️ Limitations

This project should be considered a **research/learning prototype**, not a production-grade ISL translation system.

Current limitations include:

* Alphabet-focused recognition rather than full language translation
* Dependence on webcam quality
* Sensitivity to hand visibility and positioning
* Limited training data compared with production systems
* No comprehensive sentence-level semantic translation
* Recognition performance depends heavily on the training dataset

---

# 🔮 Future Vision

The long-term goal is to move beyond isolated alphabet recognition toward a complete **Indian Sign Language communication assistant**.

```text
             Indian Sign Language
                      │
                      ▼
              Computer Vision
                      │
                      ▼
              Gesture Recognition
                      │
                      ▼
               Word Formation
                      │
                      ▼
             Sentence Generation
                      │
                      ▼
           Natural Language Processing
                      │
              ┌───────┴───────┐
              ▼               ▼
             Text            Speech
```

A future version could combine computer vision, sequence modeling and NLP to translate continuous ISL into natural-language text.

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature
```

3. Make your changes.
4. Commit your changes.

```bash
git commit -m "Add your feature"
```

5. Push the branch.

```bash
git push origin feature/your-feature
```

6. Open a Pull Request.

---

# 📜 License

Please check the repository for the applicable license before redistributing or using the project commercially.

---

# ⭐ Acknowledgements

This project uses open-source technologies including:

* OpenCV
* MediaPipe
* TensorFlow
* Keras
* NumPy

Special thanks to the open-source computer vision and machine-learning communities.

---

## 👨‍💻 Author

**HV476**

GitHub:
https://github.com/HV476

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Repository:**
https://github.com/HV476/indian-sign-language
