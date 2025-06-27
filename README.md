
---

# Human Emotion and Gesture Detector 🎭🤖

A dual-mode intelligent system that detects **human emotions** through facial expressions and identifies **hand gestures**, enabling advanced human-computer interaction. This project leverages deep learning techniques including **data augmentation**, **transfer learning** (VGG-16), and **Convolution Neural Network** to achieve high accuracy in real-time emotion and gesture recognition.

---

## 🔍 Table of Contents

* [About the Project](#About-the-Project)
* [Project Architecture](#Project-Architecture)
* [Dataset and Preprocessing](#dataset-and-preprocessing)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Model Development](#model-development)

  * [Emotion Recognition Models](#emotion-recognition-models)
  * [Gesture Recognition Model](#gesture-recognition-model)
* [Performance](#performance)
* [Code Overview](#code-overview)
* [How to Run](#how-to-run)
* [Dependencies](#dependencies)
* [License](#license)
* [Contribution](#Contribution)

---

## About the Project

The **Human Emotion and Gesture Detector** is a computer vision project that recognizes facial emotions and hand gestures either **independently or simultaneously**. The system is capable of processing real-time video feeds and predicting the corresponding emotional or gestural state of the user.

✅ **Accuracy: 87% on Emotion Recognition**
📦 Based on robust models with **VGG-16** and **Augmented CNNs**.

---

## Project Architecture 
### 🏗️
```text
fer2013.csv (Facial Emotion Dataset)
        │
        ├──> Data Preprocessing
        │        ├── Image Extraction
        │        └── Normalization
        │
        ├──> EDA (Emotion + Gesture)
        │
        ├──> Model Training
        │       ├── emotions_train (Base Emotion Model)
        │       ├── gestures_train (VGG-16 Gesture Model)
        │       └── emotions_final (Improved Emotion Model)
        │
        └──> Run Modules
                 ├── final_run.py  (Emotion OR Gesture)
                 └── final_run1.py (Emotion AND Gesture)
```

---

## Dataset and Preprocessing
### 📊
### 🔹 Emotion Dataset

* **Source**: [FER2013 - Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
* **Size**: \~35,000 labeled 48x48 grayscale facial images
* **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 🔹 Preprocessing Steps:

* Converted CSV pixel data to image arrays
* Normalized pixel values to \[0, 1] range
* One-hot encoding of emotion labels
* Data augmentation (rotation, flipping, zooming) to improve generalization

---

## Exploratory Data Analysis (EDA)
### 📈
Detailed EDA was performed to understand:

* Distribution of emotion categories
* Class imbalance analysis
* Gesture class mapping and visualizations
* Sample visual previews of both emotion and gesture datasets
* Heatmaps and confusion matrices post-training

EDA scripts are located in the `eda/` section (or embedded within Jupyter notebooks).

---

## Model Development
### 🧠
### Emotion Recognition Models
### 🎭
1. **emotions\_train.py**

   * Baseline CNN with data augmentation
   * Relu, MaxPooling, Dropout layers
   * Achieved \~82% validation accuracy

2. **emotions\_final.py**

   * Enhanced architecture with improved regularization
   * Optimizer tuned (Adam with learning rate decay)
   * Final accuracy: **87%**

### Gesture Recognition Model
### ✋
* **gestures\_train.py**

  * Based on **VGG-16 pretrained model**
  * Custom dense layers added for gesture classification
  * Trained on a labeled dataset of hand gestures
  * Transfer learning helps speed up convergence

---

## Performance
### 📊
| Model           | Accuracy | Method             |
| --------------- | -------- | ------------------ |
| emotions\_train | \~82%    | CNN + Augmentation |
| emotions\_final | **87%**  | Enhanced CNN       |
| gestures\_train | \~85%    | VGG-16 + Custom FC |

---

## Code Overview
### 🧾
| File Name               | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `data_preprocessing.py` | Converts FER2013 CSV into usable image format      |
| `emotions_train.py`     | Trains baseline emotion CNN                        |
| `emotions_final.py`     | Final optimized emotion model                      |
| `gestures_train.py`     | Trains gesture model using VGG-16                  |
| `recordings.py`         | Contains video capture and detection code          |
| `final_run.py`          | Run either emotion or gesture detection            |
| `final_run1.py`         | Run both emotion and gesture detection in parallel |

---

## How to Run
### 🧪
### ⚙️ Step-by-step Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Human-Emotion-and-Gesture-Detector.git
   cd Human-Emotion-and-Gesture-Detector
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models (optional)**

   ```bash
   python emotions_final.py
   python gestures_train.py
   ```

4. **Run Detection**

   * **Emotion or Gesture (choose one)**

     ```bash
     python final_run.py
     ```
   * **Simultaneous Emotion + Gesture Detection**

     ```bash
     python final_run1.py
     ```

---

## Dependencies
### 📦
* Python 3.8+
* OpenCV
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

Install all using:

```bash
pip install opencv-python tensorflow numpy matplotlib scikit-learn
```

---

## License
Copyright (c) 2024 Shrutayu Wankhade

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contribution
### 🤝
Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

---
