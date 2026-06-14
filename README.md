Computer Vision Project — 6th Semester
Field	Detail
Submitted By::	Shoaib Mustafa (F23BDOCS1E02071)
                Noor Elahi (F23BDOCS1E02069)   
                Taimoor Azhar (F23BDOCS1E02082)
Submitted To::	Sir Abdullah Soomro
Section::	2-E
# 🚗 License Plate Recognition (LPR) System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7%2B-orange)](https://github.com/JaidedAI/EasyOCR)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A fully-functional, end-to-end **License Plate Recognition** system built with OpenCV and EasyOCR. Features a beautiful dark-themed **Streamlit web UI**, a modular Python pipeline, a command-line interface, and step-by-step Jupyter notebook.

---

## 🎬 Demo

```
Input → Preprocessing → Detection → Enhancement → OCR → Result
  🚗        🔲📐            🟩           ✨           🧠    📋
```

**Example Output:**
```
[LPR] Plate detected at (280,310)  size 130×38
[LPR] OCR result  : 'ABC 1234'  (confidence: 87.3%)
[LPR] Total time  : 3.241s
```

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Web App](#-web-app-streamlit)
- [Command-Line Interface](#-command-line-interface)
- [Pipeline Explained](#-pipeline-explained)
- [Jupyter Notebook](#-jupyter-notebook)
- [Configuration](#-configuration)
- [Extending the Project](#-extending-the-project)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Plate Detection** | Contour-based detection with aspect-ratio filtering & fallback |
| 🧠 **OCR Engine** | EasyOCR with confidence scoring and post-processing |
| ✨ **Image Enhancement** | CLAHE + adaptive thresholding for low-light plates |
| 🔄 **Deskewing** | Automatic rotation correction |
| 🌐 **Web Interface** | Beautiful dark-themed Streamlit app |
| 📸 **Camera Input** | Real-time capture from webcam |
| ⬇️ **Export Results** | Download annotated images |
| 🖥️ **GPU Support** | CUDA acceleration for EasyOCR |
| 📊 **Pipeline Viz** | Step-by-step visualisation of every stage |
| 🔧 **CLI Tool** | Full command-line interface |

---

## 📁 Project Structure

```
license_plate_recognition/
│
├── app.py                     ← 🌐 Streamlit web application
├── lpr_pipeline.py            ← 🔧 Core pipeline + CLI entry point
├── generate_sample.py         ← 🖼️  Synthetic test image generator
├── test_pipeline.py           ← ✅ Automated pipeline test
├── requirements.txt           ← 📦 Python dependencies
├── .gitignore                 ← 🚫 Files to exclude from Git
│
├── utils/
│   ├── __init__.py            ← Package exports
│   ├── preprocessing.py       ← Grayscale, blur, Canny, CLAHE
│   ├── detection.py           ← Contour detection & annotation
│   └── recognition.py        ← EasyOCR wrapper + post-processing
│
├── notebooks/
│   └── lpr_notebook.ipynb    ← 📓 Step-by-step tutorial notebook
│
├── sample_images/             ← Test images (add your own here)
│   └── test_car.jpg           ← Auto-generated synthetic image
│
└── outputs/                   ← Pipeline results saved here
    ├── test_car_result.jpg
    └── pipeline_visualisation.jpg
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/license_plate_recognition.git
cd license_plate_recognition
```

### Step 2 — Create a virtual environment *(recommended)*

```bash
# Windows
python -m venv lpr_env
lpr_env\Scripts\activate

# macOS / Linux
python -m venv lpr_env
source lpr_env/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR will automatically download the required model files (~100 MB) on the first run.

> **GPU Support (optional):** For CUDA acceleration, install PyTorch with CUDA first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🚀 Quick Start

### Generate a test image and run the pipeline

```bash
python test_pipeline.py
```

This will:
1. Generate a synthetic car image in `sample_images/`
2. Run the complete LPR pipeline
3. Save the annotated result to `outputs/`
4. Print the recognised plate number

---

## 🌐 Web App (Streamlit)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### App Features:
- **📁 Upload Image** — Upload any car photo (JPG, PNG, WebP)
- **📸 Camera** — Capture directly from your webcam
- **🔬 Pipeline View** — See every processing stage side-by-side
- **⚙️ Sidebar Controls** — Tune Canny thresholds, languages, GPU settings
- **⬇️ Download** — Export the annotated result image

---

## 💻 Command-Line Interface

```bash
# Basic usage
python lpr_pipeline.py --image sample_images/test_car.jpg

# With GPU acceleration
python lpr_pipeline.py --image car.jpg --gpu

# Custom output directory and pipeline visualisation
python lpr_pipeline.py --image car.jpg --output-dir results --visualize

# Multi-language OCR
python lpr_pipeline.py --image car.jpg --languages en fr de

# All options
python lpr_pipeline.py --help
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | *(required)* | Path to input image |
| `--gpu` | False | Enable GPU/CUDA for OCR |
| `--languages` | en | OCR language codes |
| `--output-dir` | outputs | Directory for saved results |
| `--visualize` | False | Save pipeline visualisation |
| `--quiet` | False | Suppress console output |

---

## 🔬 Pipeline Explained

### Stage 1 — Image Preprocessing

```python
gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged   = cv2.Canny(blurred, low=50, high=150)
```

- **Grayscale** — Reduces data to 1 channel
- **Gaussian Blur** — Removes high-frequency noise
- **Canny Edges** — Highlights strong gradients (plate boundaries)

### Stage 2 — Plate Detection

```python
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Filter by: 4 corners + aspect ratio 1.5–7.0 + area constraints
```

The detector iterates over the top-K largest contours and selects the first 4-sided polygon matching license plate proportions.

### Stage 3 — Plate Enhancement

```python
upscaled  = cv2.resize(plate, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced  = clahe.apply(gray)
thresh    = cv2.adaptiveThreshold(enhanced, 255, ADAPTIVE_THRESH_GAUSSIAN_C, ...)
```

### Stage 4 — OCR

```python
reader = easyocr.Reader(['en'])
result = reader.readtext(plate_enhanced)
```

---

## 📓 Jupyter Notebook

```bash
pip install jupyter
jupyter notebook notebooks/lpr_notebook.ipynb
```

The notebook walks through every stage interactively with visualisations at each step.

---

## ⚙️ Configuration

### Canny Thresholds

| Scenario | Low | High |
|----------|-----|------|
| Standard bright image | 50 | 150 |
| High-noise background | 80 | 200 |
| Low-contrast plate | 30 | 100 |
| Overexposed image | 60 | 180 |

### Detection Sensitivity

Adjust `top_k` (default: 30) to examine more/fewer contour candidates.

---

## 🔧 Extending the Project

### 1. Add YOLOv8 Detection

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model(image)
```

### 2. Real-time Video Processing

```python
import cv2
cap = cv2.VideoCapture(0)  # 0 = webcam
while True:
    ret, frame = cap.read()
    result = run_pipeline_frame(frame)   # adapt run_pipeline for frames
    cv2.imshow("LPR", result["annotated"])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

### 3. Switch to PaddleOCR

```bash
pip install paddleocr paddlepaddle
```

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr(plate_image, cls=True)
```

---

## 🐛 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| No plate detected | Canny thresholds too strict | Reduce thresholds in sidebar |
| Plate detected, no text | EasyOCR models not downloaded | Run once with internet access |
| Wrong characters | Blurry/small plate image | Ensure plate > 80×20 px |
| Slow OCR | CPU mode | Enable GPU in sidebar, or use PaddleOCR |
| `ModuleNotFoundError` | Missing packages | Re-run `pip install -r requirements.txt` |
| `cv2.error` on imread | Unsupported image format | Convert image to JPG/PNG first |

---

## 📊 Performance

| Stage | Typical Time |
|-------|-------------|
| Load & decode | ~5 ms |
| Preprocessing | ~15 ms |
| Detection | ~20 ms |
| Enhancement | ~10 ms |
| OCR (CPU) | ~2–5 sec |
| OCR (GPU) | ~200–500 ms |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) by Jaided AI
- [OpenCV](https://opencv.org/) — Computer vision library
- [Streamlit](https://streamlit.io/) — Web framework
- Inspired by the OpenCV contour-based plate detection approach

---

*Built as a Computer Vision learning project — feel free to fork, extend, and contribute!* 🚀
