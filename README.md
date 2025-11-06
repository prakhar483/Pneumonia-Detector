# 🩺 Pneumonia Detection from Chest X-rays using CNN & Streamlit  

This project was developed as part of my **learning journey at Innomatics Research Labs**.  
It focuses on detecting **Pneumonia from Chest X-ray images** using **Convolutional Neural Networks (CNN)**  
and a simple **Streamlit web app** for real-time predictions.  

---

## 🚀 Project Overview  

The model classifies chest X-ray images into two categories:  
- **Normal** – Healthy lungs  
- **Pneumonia** – Infected lungs  

It uses deep learning techniques for image-based diagnosis and an interactive UI where users can upload their X-ray images to get instant predictions.  

---

## 🧠 Key Features  

- 🧩 **Custom CNN architecture** trained on the *Chest X-ray (Pneumonia) Dataset*  
- 🔁 **Image preprocessing & augmentation** (resizing, normalization, rotation, flipping)  
- 📊 Model evaluation with **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
- 🔍 **Grad-CAM** visualization for model explainability  
- 🌐 **Streamlit web app** for easy, interactive predictions  

---

## 📂 Repository Structure  

- ├── 📘 CNN_Project_1_on_Chest_Xray_Pneumonia_Detection.ipynb # Model training & evaluation notebook
- ├── 💻 Chest_xray_app.py # Streamlit app frontend
- ├── 🖼️ chest_app_sample_image.jpg # Sample chest X-ray image
- └── 📄 README.md # Project documentation


> ⚠️ **Note:** The trained model file (`.h5` or `.pkl`) is not uploaded to GitHub because it exceeds the 25MB size limit.  
> You can train your own model using the provided notebook and save it locally to run the app.

---

## 🧪 Dataset  

Dataset used: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  

- Total images: ~5,863  
- Classes: *Normal* and *Pneumonia*  
- Format: JPEG images resized to 150×150 / 224×224  

---
