# AI Sign Language Translator

An AI-powered Sign Language Translator that detects hand gestures and converts them into text in real time using Computer Vision and Deep Learning.

## Features

* Real-time hand gesture detection
* Sign language to text conversion
* Deep learning based gesture recognition
* OpenCV-powered webcam integration
* TensorFlow/Keras model support
* Dataset collection scripts included
* Supports custom training datasets

---

# Project Structure

```bash
AI_sign_language_translator/
│
├── Data/                        # Dataset folder
├── Model/                       # Trained model files
│   ├── keras_model.h5
│   └── labels.txt
│
├── converted_keras/             # Converted model files
├── uncertain_images/            # Uncertain prediction images
│
├── dataCollection.py            # Dataset collection script
├── dataCollection_part2.py      # Additional dataset collection
│
├── P.py                         # Prediction script
├── P2.py
├── P3.py
├── P4.py
├── P5.py
├── P7.py
├── P8.py
├── P9.py
│
└── README.md
```

---

# Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* cvzone
* NumPy

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Mayank200301/AI_sign_language_translator.git
```

## 2. Move into Project Folder

```bash
cd AI_sign_language_translator
```

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the prediction script:

```bash
python P.py
```

Or run other prediction versions:

```bash
python P2.py
python P3.py
```

---

# Dataset Collection

To collect custom hand gesture images:

```bash
python dataCollection.py
```

---

# Model Training

You can train your own gesture recognition model using TensorFlow/Keras and replace:

```bash
Model/keras_model.h5
```

with your trained model.

---

# Future Improvements

* Sentence formation
* Speech output
* Multi-language support
* Dynamic gesture recognition
* Offline AI assistant integration

---

# Author

Mayank Uikey

GitHub: https
