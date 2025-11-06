import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import cv2

st.set_page_config(page_title="🩺 Chest X-ray Pneumonia Detection", layout="centered")

# --- Sidebar: Project & Data Description ---
with st.sidebar:
    st.title("🩺 Project Info")
    st.markdown("""
    ## 📝 Problem Statement  
    Detect pneumonia from chest X-ray images using **Convolutional Neural Networks (CNNs)** and **Transfer Learning** models, with added **explainability** to highlight the critical lung regions influencing the predictions.  

    **Project involves:**  
    - Preprocessing and augmenting the Chest X-ray dataset (resizing, normalization, rotation, flipping)  
    - Building and training deep learning models (Custom CNN, ResNet, DenseNet, VGG)  
    - Evaluating performance with metrics such as **Accuracy, Precision, Recall, F1-score, and ROC-AUC**  
    - Using **Grad-CAM / Saliency maps** to visualize the regions of the lungs that contribute most to the model’s decision  
    - Developing a robust, explainable pneumonia detection system that can assist radiologists in medical diagnosis  

    ---

    ## 📂 Data Source  
    - **Chest X-ray (Pneumonia) Dataset** by Guangzhou Women and Children’s Medical Center  
    - [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
    - Labeled as **Normal** or **Pneumonia**  

    ---

    ## 📊 Dataset Overview  
    - **Total images:** ~5,863 X-ray scans  
    - **Classes:** Normal (Healthy Lungs), Pneumonia (Infected Lungs)  
    - **Data Split:**  
        - Training: 5,216 images  
        - Validation: 16% split from training  
        - Testing: 624 images  
    - **Image Format:** Grayscale/RGB JPEG, resized to 150x150 or 224x224  

    ---

    ## 🎯 Target Labels  
    - **Normal:** No signs of infection  
    - **Pneumonia:** Symptoms of bacterial or viral pneumonia  

    ---

    ## 🔑 Key Highlights  
    - Explainable CNN-based pneumonia detection system  
    - Transfer Learning (ResNet, DenseNet, VGG)  
    - Grad-CAM for interpretability  
    - **X% accuracy, Y% F1-score** on test data  
    - Demonstrates AI-assisted radiology in healthcare diagnostics  
    """)

    st.markdown("---")
    st.markdown("### 🖼️ Sample Image for Testing")
    st.markdown(
        """
        [Download Sample Normal X-ray](https://storage.googleapis.com/kagglesdsdata/datasets/3137/5242/chest_xray/test/NORMAL/IM-0001-0001.jpeg)
        """
    )
    st.caption("Right-click the link above and choose 'Save link as...' to download a sample image for testing.")

# --- App Title and Description ---
st.title("🩺 Chest X-ray Pneumonia Detection")
st.markdown("""
Detect **Pneumonia** from chest X-ray images using a trained Convolutional Neural Network (CNN).  
Upload a chest X-ray image and get an instant prediction with explainability (Grad-CAM heatmap).
""")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("simple_cnn_chest_xray.h5")
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(img, target_size=(150, 150)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Grad-CAM Implementation ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = np.array(img.resize((heatmap.shape[1], heatmap.shape[0])))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a Chest X-ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_container_width=True)
    img_array = preprocess_image(img)

    # --- Prediction ---
    prediction = model.predict(img_array)[0][0]
    pred_label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: **{pred_label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    # --- Grad-CAM Explainability ---
    try:
        # You may need to change 'conv2d_2' to the last conv layer name in your model
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2")
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, img.size)
        overlayed_img = overlay_heatmap(img, heatmap_resized)
        st.markdown("#### 🔍 Model Focus (Grad-CAM Heatmap):")
        st.image(overlayed_img, caption="Grad-CAM Heatmap", use_container_width=True)
    except Exception as e:
        st.warning("Grad-CAM visualization not available for this model architecture.")
        st.text(str(e))

    st.markdown("---")
    st.info("**Disclaimer:** This tool is for educational/demo purposes only. Not for clinical use.")

else:
    st.info("Please upload a chest X-ray image to get a prediction.")

# --- Footer ---
st.markdown(
    """
    <hr>
    <center>
    <small>
    Developed with ❤️ by Prakhar Dwivedi for AI-assisted radiology.<br>
    Dataset: <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" target="_blank">Chest X-ray (Pneumonia)</a>
    </small>
    </center>
    """,
    unsafe_allow_html=True,
)