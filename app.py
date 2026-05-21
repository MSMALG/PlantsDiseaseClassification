import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# Page config 
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Load all models and assets (cached so they load once) 
@st.cache_resource
def load_dl_model():
    return tf.keras.models.load_model("best_model.keras")

@st.cache_resource
def load_classical_assets():
    return {
        "svm":    joblib.load("svm_model.pkl"),
        "rf":     joblib.load("rf_model.pkl"),
        "scaler": joblib.load("hog_scaler.pkl"),
        "pca":    joblib.load("hog_pca.pkl"),
        "le":     joblib.load("label_encoder.pkl"),
    }

@st.cache_data
def load_meta():
    return {
        "class_names": joblib.load("class_names.pkl"),
        "dl_acc":      joblib.load("dl_accuracy.pkl"),
        "svm_acc":     joblib.load("svm_accuracy.pkl"),
        "rf_acc":      joblib.load("rf_accuracy.pkl"),
    }

meta    = load_meta()
assets  = load_classical_assets()
dl_model = load_dl_model()

IMG_SIZE = 224
HOG_SIZE = 64

# HOG feature extraction 
def get_hog_features(pil_img):
    """Extract HOG features from a PIL image — same pipeline as training."""
    img  = pil_img.convert("RGB").resize((HOG_SIZE, HOG_SIZE))
    gray = rgb2gray(np.array(img))
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    feat_scaled = assets["scaler"].transform([feat])
    feat_pca    = assets["pca"].transform(feat_scaled)
    return feat_pca

# Prediction function 
def predict(pil_img, model_key):
    if model_key == "efficientnet":
        # Use EfficientNet's own preprocessing — NOT /255
        # Must match exactly what was used during training
        img_arr = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        probs   = dl_model.predict(img_arr, verbose=0)[0]
        idx     = np.argmax(probs)
        label   = meta["class_names"][idx]
        conf    = probs[idx]
    else:
        feat  = get_hog_features(pil_img)
        clf   = assets["svm"] if model_key == "svm" else assets["rf"]
        idx   = clf.predict(feat)[0]
        probs = clf.predict_proba(feat)[0]
        label = assets["le"].inverse_transform([idx])[0]
        conf  = probs[idx]
    return label, conf

# Parse class label into plant and condition
def parse_label(label):
    parts = label.replace("___", "__").split("__")
    plant = parts[0].replace("_", " ").title() if len(parts) > 0 else "Unknown"
    cond  = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return plant, cond


# UI Layout
st.title("🌿 Plant Disease Detector")
st.markdown(
    "Upload a leaf image and select a model. "
    "The app will classify the plant disease or confirm it is healthy."
)

st.divider()

#  Model selector 
st.subheader("1 · Choose a Model")

MODEL_OPTIONS = {
    f"🧠 EfficientNetB0  —  Accuracy: {meta['dl_acc']:.2%}  [Deep Learning]":  "efficientnet",
    f"📐 SVM (HOG)       —  Accuracy: {meta['svm_acc']:.2%}  [Classical ML]":  "svm",
    f"🌲 Random Forest   —  Accuracy: {meta['rf_acc']:.2%}  [Classical ML]":   "rf",
}

chosen_label = st.radio("Select model:", list(MODEL_OPTIONS.keys()), index=0)
model_key    = MODEL_OPTIONS[chosen_label]

# Description of selected model
descriptions = {
    "efficientnet": (
        "**EfficientNetB0** is a convolutional neural network pretrained on ImageNet "
        "and fine-tuned on PlantVillage. It learns visual features automatically from pixels. "
        "Best accuracy but requires more compute."
    ),
    "svm": (
        "**SVM** uses HOG (Histogram of Oriented Gradients) — handcrafted edge and shape "
        "descriptors — combined with a Support Vector Machine classifier. "
        "Faster but less accurate than deep learning."
    ),
    "rf": (
        "**Random Forest** also uses HOG features but classifies with an ensemble of "
        "decision trees voting on the final prediction. "
        "Fastest but lowest accuracy on this dataset."
    ),
}
st.info(descriptions[model_key])

st.divider()

# Image upload 
st.subheader("2 · Upload a Leaf Image")
uploaded = st.file_uploader(
    "Choose a leaf image (.jpg or .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    st.divider()
    st.subheader("3 · Prediction")

    with st.spinner("Analyzing leaf …"):
        label, confidence = predict(pil_img, model_key)

    plant, condition = parse_label(label)
    is_healthy = "healthy" in condition.lower()

    # Result display
    if is_healthy:
        st.success(f"✅ **{plant}** — Healthy")
    else:
        st.error(f"⚠️ **{plant}** — {condition}")

    st.metric("Confidence", f"{confidence:.1%}")

    with st.expander("Raw class label"):
        st.code(label)

    st.divider()
    st.caption(
        "⚠️ Note: This model was trained on lab-condition images (PlantVillage dataset). "
        "Predictions on outdoor photos may be less reliable due to background, "
        "lighting, and angle differences — a known limitation called domain shift."
    )