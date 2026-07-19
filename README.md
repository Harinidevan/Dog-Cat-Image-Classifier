# 🐶🐱 Dog & Cat Image Classifier

A Deep Learning-based Dog & Cat Image Classifier built using **TensorFlow**, **MobileNetV2**, and **Streamlit**. The application classifies uploaded images as either **Dog** or **Cat** with high accuracy through an interactive web interface.

---

## 📌 Project Overview

This project uses **Transfer Learning** with the **MobileNetV2** pre-trained model to perform binary image classification. Users can upload an image through a Streamlit web application, and the model predicts whether the image is a **Dog** or a **Cat** along with the prediction confidence.

---

## 🚀 Features

- 🐶 Classifies images as Dog or Cat
- 📤 Upload images through a Streamlit web application
- 📊 Displays prediction confidence
- 🧠 Transfer Learning using MobileNetV2
- ⚡ Fast and accurate predictions
- 💾 Trained model included

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit

---

## 📂 Project Structure

```
Dog-Cat-Image-Classifier/
│
├── app.py
├── train.py
├── predict.py
├── reduce_dataset.py
├── requirements.txt
├── README.md
│
└── models/
    └── dog_cat_classifier.keras
```

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Test Accuracy | **98.40%** |
| Test Loss | **0.0507** |

---

## 📥 Dataset

This project uses the **Dogs vs Cats** image dataset.

**Dataset Link:**
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

> **Note:** The dataset is not included in this repository because of its large size.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Harinidevan/Dog-Cat-Image-Classifier.git
```

### Navigate to the project folder

```bash
cd Dog-Cat-Image-Classifier
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖼️ How to Use

1. Launch the Streamlit application.
2. Click **Browse Files**.
3. Upload a Dog or Cat image.
4. View the prediction and confidence score.

---

## 🧠 Deep Learning Workflow

- Dataset Preparation
- Image Preprocessing
- Data Augmentation
- Transfer Learning using MobileNetV2
- Model Training
- Model Evaluation
- Prediction
- Streamlit Web Application

---

## 📈 Future Improvements

- Support for multiple animal classes
- Deploy the application online
- Improve UI with additional visualizations
- Real-time webcam prediction

---

## 👩‍💻 Author

**Harini D**

B.Tech Artificial Intelligence & Data Science

---

## ⭐ If you found this project useful, please consider giving it a star!# Dog-Cat-Image-Classifier
A Deep Learning-based Dog &amp; Cat Image Classifier built using TensorFlow, MobileNetV2, and Streamlit to accurately classify uploaded images as dogs or cats through an interactive web application.
