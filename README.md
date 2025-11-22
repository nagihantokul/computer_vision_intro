# 🖼️ Basic Computer Vision App using OpenCV & Streamlit

This repository contains a beginner‑friendly but fully functional **Computer Vision application** built with **Python, OpenCV, and Streamlit**. The project demonstrates how classic CV techniques can be combined with an interactive UI to create a simple image‑processing tool.

The project includes:

- `cv.py` → Main Streamlit application for image processing  
- `computer_vision_intro.ipynb` → Jupyter Notebook for testing and experimenting with OpenCV functions  

---

## 📁 Project Structure

```
.
├── cv.py                # Streamlit app with multiple CV operations
├── computer_vision_intro.ipynb            # Notebook for experimentation and demonstrations
└── README.md
```

---

## 🚀 Features

The app supports a range of image transformations and computer vision tasks:

### ✔ Supported Operations
- **Original Image**
- **Face Detection**
- **Edge Detection**
- **Grayscale**
- **Blur**
- **Thresholding**
- **Contour Detection**
- **Sharpening**
- **Emboss Effect**
- **Cartoon Effect**
- **Invert Colors**

### ✔ Key Technologies
- **OpenCV** for all computer vision operations  
- **NumPy** for array manipulation  
- **Pillow** for image loading  
- **Streamlit** for interactive UI  

---

## 🎯 How the App Works

### 1️⃣ Load an Image
The user uploads an image using Streamlit’s file uploader.

### 2️⃣ Choose an Operation
A dropdown menu lists all available OpenCV-based effects.

### 3️⃣ Process the Image
The selected operation is applied automatically upon clicking **Apply**.

### 4️⃣ Download the Processed Image  
Users can download their edited output as a `.png` file.

---

## 🧠 Core Functions Overview

### 🔍 Face Detection  
Uses OpenCV’s Haar Cascade classifier:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

### ✏️ Edge Detection  
Implemented with **Canny Edge Detector**:

```python
edges = cv2.Canny(gray, 100, 200)
```

### 🟩 Contour Detection  
Contours are detected after thresholding:

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

### 🎨 Cartoonize Effect  
Uses bilateral filtering + edge masking.

---

## 🖥️ Running the App Locally

### 1️⃣ Install Dependencies  
Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install streamlit opencv-python pillow numpy
```

---

### 2️⃣ Run the Streamlit App

```bash
streamlit run cv.py
```

Your browser will open with the full application UI.

---

## 📓 Jupyter Notebook (`cv2.ipynb`)
This notebook includes:

- Experiments with individual OpenCV functions  
- Demonstrations of filters such as blur, sharpen, emboss  
- Visualizations of intermediate steps (grayscale → edges → contours)

It serves as a sandbox for learning core CV concepts before porting them into the Streamlit app.

---

## ✨ Possible Enhancements

- Add webcam support  
- Add real‑time video filters  
- Include segmentation models (U‑Net, DeepLab)  
- Add object detection (YOLO, SSD, Faster R‑CNN)  
- Deploy to Streamlit Cloud  

---

## 🤝 Contributions  
Feel free to fork the repository, open issues, or submit pull requests.

---

## 📜 License  
This project can use MIT License or any license of your choice.

