# MNIST Handwritten Digit Recognition using Deep CNN

## 📌 Project Overview
This project implements a **Deep Convolutional Neural Network (CNN)** for **handwritten digit recognition** using the **MNIST dataset**. The model is designed to accurately classify digits (0-9) using deep learning techniques, enhancing real-world automation applications such as:

- 📮 **Postal Mail Sorting**
- 🏦 **Bank Check Processing**
- 📑 **Automated Form Digitization**

## 🎯 Problem Statement
Traditional methods struggle with handwritten digit recognition due to:
- ✍ **Variability in Handwriting Styles** – Different individuals write digits uniquely.
- 🌀 **Noisy or Distorted Inputs** – Blurred, incomplete, or overlapping digits reduce accuracy.
- ⚖ **Scalability Issues** – Traditional algorithms fail to generalize across diverse datasets.

## 🚀 Objective
The goal of this project is to develop a **Deep CNN model** that:
- ✅ Improves recognition efficiency with minimal manual intervention.
- ✅ Enhances pattern detection through deep learning techniques.
- ✅ Ensures robust generalization across diverse handwriting samples.

## 📖 Abstract
We implemented a **Deep CNN** using **TensorFlow and Keras** to classify handwritten digits with high accuracy. The architecture leverages:
- 🏗 **Convolutional Layers** – Extract spatial features from digit images.
- 🎯 **Pooling Layers** – Reduce dimensionality while preserving key patterns.
- 🏁 **Fully Connected Layers** – Perform classification based on learned features.

## 📂 Project Structure
```
📂 MNIST-Digit-Recognition
│── 📜 README.md
│── 📂 data
│── 📂 notebooks
│── 📂 models
│── 📂 src
│── 📄 train.py
│── 📄 predict.py
│── 📄 requirements.txt
```

## 🛠️ Implementation Steps
### 1️⃣ Data Preparation
- Loaded the **MNIST dataset**.
- Normalized pixel values from **[0,255] → [0,1]** for faster convergence.
- Reshaped images to **(28,28,1)** for CNN input compatibility.
- One-hot encoded labels for multi-class classification.

### 2️⃣ Model Architecture
The model is based on **LeNet-5**, optimized for small-resolution images:
```
Input → [[Conv2D → ReLU] × 2 → MaxPool2D → Dropout] × 2 → Flatten → Dense → Dropout → Output
```
#### 🔹 Optimizations:
- 🌀 **Data Augmentation** – Rotating, flipping, and zooming to improve generalization.
- 🔄 **ReduceLROnPlateau** – Adjusts learning rate dynamically.
- 🚀 **RMSProp Optimizer** – Ensures stable and faster convergence.

### 3️⃣ Model Training
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** RMSProp.
- **Batch Size & Epochs:** Tuned for optimal performance.
- **Hardware:** Trained using **GPU acceleration**.

### 4️⃣ Model Evaluation
- 📉 **Learning Curve** – Monitored training vs. validation loss.
- 📊 **Confusion Matrix** – Identified misclassification patterns.
- ✅ **Final Accuracy** – Achieved **high accuracy** on the test dataset.

### 5️⃣ Prediction on Test Data
- Generated predictions on unseen test samples.
- Stored results in a **CSV file** for further analysis.

## 📊 Managerial Insights
### ✅ **Automation Potential**
- Can be deployed in **banking, postal services, and government agencies**.
- Reduces **manual data entry errors** and speeds up processing.

### 💰 **Cost-Effectiveness**
- Replaces manual transcription with **AI-powered recognition**.
- Data augmentation minimizes the need for large datasets, reducing costs.

### 🌍 **Scalability & Adaptability**
- Can be extended for recognizing **handwritten characters in different languages**.
- Useful in **finance, healthcare, and government document processing**.

### 🚀 **Performance vs. Infrastructure**
- Requires **GPU support** for large-scale deployment.
- Can be optimized for **cloud-based AI services**.

## 📌 How to Run the Project
### 🖥️ 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 🚀 2️⃣ Train the Model
```sh
python train.py
```
### 🔍 3️⃣ Make Predictions
```sh
python predict.py --input test_data.csv --output predictions.csv
```

## 🏆 Results
- Achieved **high accuracy** in digit classification.
- Successfully deployed model for handwritten digit recognition.

## 📜 Contributors
- **Aayush Garg** (055001)
- **Saloni Gupta** (055039)

📌 **Submitted to:** Prof. Amarnath Mitra

## 📄 License
This project is licensed under the **MIT License**. Feel free to use and modify!

---
🚀 **Let's build AI-powered automation for the future!**
