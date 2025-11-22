import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------------
# Image Processing Functions
# ---------------------

def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)  # Returns RGB array by default

def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def face_detection(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return convert_bgr_to_rgb(image)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def contour_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return convert_bgr_to_rgb(image)

def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return convert_bgr_to_rgb(cv2.filter2D(image, -1, kernel))

def emboss(image):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return convert_bgr_to_rgb(cv2.filter2D(image, -1, kernel))

def cartoonize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon_image = cv2.bitwise_and(color, color, mask=edges)
    return convert_bgr_to_rgb(cartoon_image)

def invert(image):
    return convert_bgr_to_rgb(cv2.bitwise_not(image))

def process_image(image, operation):
    operations = {
        "Original": lambda img: img,  # ← DO NOT convert color for original
        "Face Detection": face_detection,
        "Edge Detection": edge_detection,
        "Grayscale": grayscale,
        "Blur": blur,
        "Threshold": threshold,
        "Contour Detection": contour_detection,
        "Sharpen": sharpen,
        "Emboss": emboss,
        "Cartoonize": cartoonize,
        "Invert": invert
    }
    return operations.get(operation, lambda img: img)(image.copy())

# ---------------------
# Streamlit App
# ---------------------

def main():
    st.set_page_config(page_title="🖼️ Image Editor", layout="wide")
    st.title("🖼️ Streamlit Image Processing App")
    st.caption("Upload an image and apply OpenCV effects instantly.")

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_file = st.file_uploader("1. Upload an Image", type=["jpg", "jpeg", "png"])
        operation = st.selectbox("2. Choose an Operation", [
            "Original", "Face Detection", "Edge Detection", "Grayscale", "Blur",
            "Threshold", "Contour Detection", "Sharpen", "Emboss", "Cartoonize", "Invert"
        ])
        apply = st.button("3. ▶️ Apply")

    if uploaded_file:
        original = load_image(uploaded_file)
        st.session_state["original_image"] = original

        if apply:
            # Convert original to BGR before processing (OpenCV expects BGR)
            image_for_opencv = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            processed = process_image(image_for_opencv, operation)
            st.session_state["processed_image"] = processed

        if "processed_image" in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖼️ Original Image")
                st.image(st.session_state["original_image"], use_column_width=True)  # RGB olduğu için dönüştürmeye gerek yok

            with col2:
                st.subheader(f"🎨 {operation} Image")
                st.image(st.session_state["processed_image"], use_column_width=True)

            st.download_button(
                label="📥 Download Processed Image",
                data=cv2.imencode(".png", cv2.cvtColor(st.session_state["processed_image"], cv2.COLOR_RGB2BGR))[1].tobytes(),
                file_name="processed_image.png",
                mime="image/png"
            )

main()
