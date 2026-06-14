"""
app.py
======
Streamlit web interface for the License Plate Recognition (LPR) system.

Run with:
    streamlit run app.py
"""

import os
import io
import time
import tempfile
import sys
from pathlib import Path
from typing import Optional

# Set standard streams to use UTF-8 to prevent Windows cp1252/UnicodeEncodeError with libraries (like EasyOCR) that print UTF-8 characters.
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

# ─── Project imports ──────────────────────────────────────────────────────────
from utils.preprocessing import preprocess_image, enhance_plate_image, deskew_plate
from utils.detection     import detect_plate, draw_detection, get_all_candidates
from utils.recognition   import recognize_text, format_raw_results

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LPR System – License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 40%, #0a1628 100%);
    min-height: 100vh;
}

/* ── Main container ── */
.main .block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a2332 0%, #162542 50%, #1a2d4a 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(59, 130, 246, 0.15);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.6rem;
    font-weight: 400;
}
.hero-badges {
    margin-top: 1.2rem;
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: #93c5fd;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #1a2540);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.2);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #60a5fa;
    line-height: 1;
}
.metric-label {
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section headers ── */
.section-header {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Result plate display ── */
.plate-display {
    background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(59,130,246,0.25), inset 0 0 20px rgba(59,130,246,0.05);
}
.plate-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #facc15;
    letter-spacing: 0.25em;
    text-shadow: 0 0 20px rgba(250,204,21,0.4);
    line-height: 1.2;
}
.plate-label {
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}

/* ── Pipeline stage cards ── */
.stage-card {
    background: #1a2332;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
}
.stage-title {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* ── Info panel ── */
.info-panel {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #93c5fd;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Warning panel ── */
.warn-panel {
    background: rgba(251, 146, 60, 0.08);
    border: 1px solid rgba(251, 146, 60, 0.3);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #fdba74;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Success panel ── */
.success-panel {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #6ee7b7;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #0a1628 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* ── Streamlit element overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.35);
}

/* ── Expanders ── */
details {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    background: #141c2b !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #7c3aed, #34d399) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(59,130,246,0.35) !important;
    border-radius: 12px !important;
    background: rgba(59,130,246,0.04) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a2332;
    border-radius: 12px;
    padding: 0.3rem;
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
}

/* ── Dividers ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1421; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def gray_to_pil(img_gray: np.ndarray) -> Image.Image:
    return Image.fromarray(img_gray)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def save_temp(uploaded_file) -> str:
    """Save an uploaded file to a temp path and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def confidence_color(conf: float) -> str:
    if conf >= 0.75:
        return "#34d399"   # green
    elif conf >= 0.45:
        return "#facc15"   # yellow
    return "#f87171"       # red


def draw_pipeline_stages(stages, result):
    """Return a list of (pil_image, label) for display."""
    panels = []
    panels.append((bgr_to_pil(stages["original"]), "① Original"))
    panels.append((gray_to_pil(stages["gray"]),     "② Grayscale"))
    panels.append((gray_to_pil(stages["blurred"]),  "③ Blur"))
    panels.append((gray_to_pil(stages["edged"]),    "④ Edges"))
    panels.append((gray_to_pil(stages["clahe"]),    "⑤ CLAHE"))
    if result.get("annotated") is not None:
        panels.append((bgr_to_pil(result["annotated"]), "⑥ Detection"))
    if result.get("plate_image") is not None:
        panels.append((bgr_to_pil(result["plate_image"]), "⑦ Crop"))
    return panels


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
            <div style="font-size:3rem;">🚗</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">LPR System</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:0.3rem;">
                License Plate Recognition
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Settings")

        use_gpu = st.toggle("🖥️ Use GPU (CUDA)", value=False,
                            help="Enable if you have an NVIDIA GPU with CUDA installed")

        languages_map = {
            "English": "en",
            "Chinese (Simplified)": "ch_sim",
            "Arabic": "ar",
            "French": "fr",
            "German": "de",
            "Hindi": "hi",
            "Japanese": "ja",
            "Korean": "ko",
            "Spanish": "es",
        }
        selected_langs = st.multiselect(
            "🌐 OCR Languages",
            list(languages_map.keys()),
            default=["English"],
            help="Select languages present on the license plate",
        )
        lang_codes = [languages_map[l] for l in selected_langs] if selected_langs else ["en"]

        st.divider()
        st.markdown("### 🔧 Detection Tuning")
        canny_low  = st.slider("Canny Low Threshold",  10,  200, 50,  step=5)
        canny_high = st.slider("Canny High Threshold", 50,  400, 150, step=10)
        top_k      = st.slider("Top Contours to Check", 5, 60,  30,  step=5)

        st.divider()
        st.markdown("### 📖 How It Works")
        st.markdown("""
        <div class="info-panel">
        <b>Two-Stage Pipeline:</b><br>
        <b>1. Detection</b> – Contour analysis finds rectangular plate regions using Canny edges.<br><br>
        <b>2. Recognition</b> – EasyOCR reads characters from the enhanced plate crop.<br><br>
        <b>Enhancement</b> – CLAHE + adaptive thresholding improve low-light plates.
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="text-align:center; color:#334155; font-size:0.75rem;">
            Built with OpenCV + EasyOCR + Streamlit
        </div>
        """, unsafe_allow_html=True)

    return use_gpu, lang_codes, canny_low, canny_high, top_k


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    use_gpu, lang_codes, canny_low, canny_high, top_k = render_sidebar()

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🔍 License Plate Recognition</h1>
        <p class="hero-subtitle">
            AI-powered vehicle license plate detection & OCR — upload an image and get results instantly.
        </p>
        <div class="hero-badges">
            <span class="badge">🧠 EasyOCR</span>
            <span class="badge">📷 OpenCV</span>
            <span class="badge">🎯 Contour Detection</span>
            <span class="badge">✨ CLAHE Enhancement</span>
            <span class="badge">🐍 Python 3.8+</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_upload, tab_camera, tab_about = st.tabs(
        ["📁 Upload Image", "📸 Use Camera", "ℹ️ About & Docs"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – UPLOAD IMAGE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_upload:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-header">📤 Upload Vehicle Image</div>',
                        unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drop a car/vehicle image here",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Works best with a clear frontal or rear view of a vehicle",
                label_visibility="collapsed",
            )

            if uploaded:
                pil_img = Image.open(uploaded)
                st.image(pil_img, caption="Uploaded image", use_container_width=True)

                st.markdown('<div class="section-header" style="margin-top:1.5rem;">📊 Image Info</div>',
                            unsafe_allow_html=True)
                info_df = pd.DataFrame({
                    "Property": ["Filename", "Format", "Dimensions", "Size"],
                    "Value": [
                        uploaded.name,
                        pil_img.format or Path(uploaded.name).suffix.upper()[1:],
                        f"{pil_img.width} × {pil_img.height} px",
                        f"{uploaded.size / 1024:.1f} KB",
                    ],
                })
                st.dataframe(info_df, hide_index=True, use_container_width=True)

                run_btn = st.button("🚀 Run LPR Pipeline", type="primary", key="run_upload")
            else:
                st.markdown("""
                <div class="info-panel" style="margin-top:1rem;">
                    <b>💡 Tips for best results:</b><br>
                    • Use a clear, well-lit image<br>
                    • Plate should be visible and not heavily blurred<br>
                    • Works with JPEG, PNG, BMP, and WebP<br>
                    • Minimum plate size: ~80×20 pixels
                </div>
                """, unsafe_allow_html=True)
                run_btn = False

        with col_right:
            st.markdown('<div class="section-header">📊 Recognition Results</div>',
                        unsafe_allow_html=True)

            if uploaded and run_btn:
                # ── Save to temp file ─────────────────────────────────────────
                uploaded.seek(0)
                tmp_path = save_temp(uploaded)

                # ── Run pipeline with progress bar ────────────────────────────
                progress = st.progress(0, text="Initialising…")
                status   = st.empty()

                try:
                    status.markdown("**⚙️ Preprocessing…**")
                    progress.progress(15)

                    image  = cv2.imread(tmp_path)
                    # Apply user-tuned Canny thresholds
                    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred= cv2.GaussianBlur(gray, (5, 5), 0)
                    edged  = cv2.Canny(blurred, canny_low, canny_high)

                    stages = preprocess_image(image)
                    stages["edged"] = edged   # override with user thresholds
                    progress.progress(30)

                    status.markdown("**🔍 Detecting plate…**")
                    plate_contour, plate_cropped, bbox = detect_plate(
                        image, edged, gray, top_k=top_k
                    )
                    progress.progress(55)

                    result = {
                        "plate_text": "",
                        "confidence": 0.0,
                        "bbox": bbox,
                        "plate_image": plate_cropped,
                        "annotated": draw_detection(image, plate_contour, bbox)
                                     if plate_contour is not None else image.copy(),
                        "raw_results": [],
                    }

                    if plate_cropped is not None:
                        status.markdown("**✨ Enhancing plate…**")
                        try:
                            plate_deskewed = deskew_plate(plate_cropped)
                            plate_enhanced = enhance_plate_image(plate_deskewed)
                        except Exception:
                            plate_enhanced = plate_cropped
                        progress.progress(70)

                        status.markdown("**🧠 Running OCR…**")
                        plate_text, confidence, raw_results = recognize_text(
                            plate_enhanced, languages=lang_codes, use_gpu=use_gpu
                        )
                        result["plate_text"]  = plate_text
                        result["confidence"]  = confidence
                        result["raw_results"] = raw_results
                        result["annotated"]   = draw_detection(
                            image, plate_contour, bbox, text=plate_text
                        )
                    progress.progress(100)
                    status.empty()
                    progress.empty()

                    # ── Store in session state ────────────────────────────────
                    st.session_state["last_result"] = result
                    st.session_state["last_stages"] = stages
                    st.session_state["last_tmp"]    = tmp_path

                except Exception as exc:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ Pipeline error: {exc}")
                    result = None

            # ── Show last result ───────────────────────────────────────────────
            if "last_result" in st.session_state:
                result = st.session_state["last_result"]

                # ── Plate display ─────────────────────────────────────────────
                if result["plate_text"]:
                    conf_pct = result["confidence"] * 100
                    col = confidence_color(result["confidence"])
                    st.markdown(f"""
                    <div class="plate-display">
                        <div class="plate-text">{result['plate_text']}</div>
                        <div class="plate-label">Recognised Plate Number</div>
                        <div style="margin-top:0.8rem; font-size:0.85rem; color:{col}; font-weight:600;">
                            Confidence: {conf_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warn-panel">
                        ⚠️ <b>No plate text recognised.</b><br>
                        Try adjusting the Canny thresholds in the sidebar, or use a clearer image.
                    </div>
                    """, unsafe_allow_html=True)

                # ── Metric row ────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                bbox = result.get("bbox")
                plate_dims = f"{bbox[2]}×{bbox[3]}" if bbox else "—"
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{'✅' if result['plate_text'] else '❌'}</div>
                        <div class="metric-label">Plate Found</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">{plate_dims}</div>
                        <div class="metric-label">Plate Size (px)</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">
                            {len(result['plate_text'].replace(' ','').replace('-',''))}
                        </div>
                        <div class="metric-label">Characters</div>
                    </div>""", unsafe_allow_html=True)

                # ── Annotated image ───────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">🖼️ Annotated Result</div>',
                            unsafe_allow_html=True)
                st.image(bgr_to_pil(result["annotated"]), use_container_width=True)

                # ── Cropped plate ─────────────────────────────────────────────
                if result["plate_image"] is not None:
                    with st.expander("🔍 View Cropped Plate"):
                        p = result["plate_image"]
                        p_big = cv2.resize(p, (p.shape[1]*3, p.shape[0]*3),
                                           interpolation=cv2.INTER_CUBIC)
                        st.image(bgr_to_pil(p_big), use_container_width=True)

                # ── Raw OCR details ───────────────────────────────────────────
                if result["raw_results"]:
                    with st.expander("📋 Raw OCR Output"):
                        fmt = format_raw_results(result["raw_results"])
                        if fmt:
                            df = pd.DataFrame(fmt)[["text", "confidence"]]
                            df.columns = ["Detected Text", "Confidence (%)"]
                            st.dataframe(df, hide_index=True, use_container_width=True)

                # ── Download button ───────────────────────────────────────────
                annotated_bgr = result["annotated"]
                _, img_enc = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                st.download_button(
                    "⬇️ Download Result Image",
                    data=img_enc.tobytes(),
                    file_name="lpr_result.jpg",
                    mime="image/jpeg",
                )

            else:
                st.markdown("""
                <div class="info-panel">
                    👈 Upload an image and click <b>Run LPR Pipeline</b> to get started.
                </div>
                """, unsafe_allow_html=True)

        # ── Pipeline visualisation ─────────────────────────────────────────────
        if "last_stages" in st.session_state and "last_result" in st.session_state:
            st.divider()
            st.markdown('<div class="section-header">🔬 Pipeline Visualisation — Step by Step</div>',
                        unsafe_allow_html=True)
            panels = draw_pipeline_stages(
                st.session_state["last_stages"],
                st.session_state["last_result"],
            )
            cols = st.columns(len(panels))
            for col, (pil_img, label) in zip(cols, panels):
                with col:
                    st.markdown(f'<div class="stage-title">{label}</div>',
                                unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – CAMERA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_camera:
        st.markdown("""
        <div class="section-header">📸 Capture from Camera</div>
        """, unsafe_allow_html=True)

        camera_img = st.camera_input(
            "Take a photo of the vehicle's license plate",
            label_visibility="collapsed",
        )

        if camera_img:
            pil_cam = Image.open(camera_img)
            img_bgr = pil_to_bgr(pil_cam)

            with st.spinner("🧠 Processing…"):
                gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edged   = cv2.Canny(blurred, canny_low, canny_high)
                stages  = preprocess_image(img_bgr)
                stages["edged"] = edged

                plate_contour, plate_cropped, bbox = detect_plate(
                    img_bgr, edged, gray, top_k=top_k
                )
                plate_text, confidence, raw_results = ("", 0.0, [])

                if plate_cropped is not None:
                    try:
                        plate_enhanced = enhance_plate_image(deskew_plate(plate_cropped))
                    except Exception:
                        plate_enhanced = plate_cropped
                    plate_text, confidence, raw_results = recognize_text(
                        plate_enhanced, languages=lang_codes, use_gpu=use_gpu
                    )

                annotated = draw_detection(img_bgr, plate_contour, bbox, text=plate_text) \
                            if plate_contour is not None else img_bgr.copy()

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(bgr_to_pil(annotated), caption="Detection Result", use_container_width=True)
            with col_b:
                if plate_text:
                    st.markdown(f"""
                    <div class="plate-display">
                        <div class="plate-text">{plate_text}</div>
                        <div class="plate-label">Recognised Plate</div>
                        <div style="margin-top:0.8rem; color:{confidence_color(confidence)}; font-weight:600;">
                            Confidence: {confidence*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warn-panel">
                        ⚠️ No plate detected. Make sure the plate is visible and well-lit.
                    </div>
                    """, unsafe_allow_html=True)

                if plate_cropped is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">🔍 Cropped Plate</div>',
                                unsafe_allow_html=True)
                    p = plate_cropped
                    p_big = cv2.resize(p, (p.shape[1]*3, p.shape[0]*3),
                                       interpolation=cv2.INTER_CUBIC)
                    st.image(bgr_to_pil(p_big), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – ABOUT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_about:
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            st.markdown("""
            ## 🚗 License Plate Recognition System

            This project implements a **two-stage computer vision pipeline** for automatic
            license plate detection and character recognition.

            ### 🔬 Pipeline Architecture

            ```
            Input Image
                ↓
            Grayscale Conversion
                ↓
            Gaussian Blur (noise reduction)
                ↓
            Canny Edge Detection
                ↓
            Contour Analysis → Plate Detection
                ↓
            CLAHE Enhancement + Deskew
                ↓
            Adaptive Thresholding
                ↓
            EasyOCR → Text Extraction
                ↓
            Post-processing → Final Plate Number
            ```

            ### 📦 Tech Stack

            | Component        | Library / Tool          |
            |-----------------|-------------------------|
            | Image Processing | OpenCV 4.8+             |
            | OCR Engine       | EasyOCR 1.7+            |
            | Array Operations | NumPy                   |
            | Web Interface    | Streamlit 1.28+         |
            | Image Utilities  | Pillow                  |
            | Data Display     | Pandas                  |

            ### 🎯 Detection Algorithm

            1. **Edge Detection** – Canny edge detector highlights plate boundaries
            2. **Contour Finding** – OpenCV extracts all closed contours from edge map
            3. **Shape Filtering** – Contours are filtered by:
               - Area (removes noise)
               - 4-sided polygon shape
               - Aspect ratio (1.5 – 7.0 for standard plates)
            4. **Fallback** – If no 4-sided polygon is found, the largest aspect-matching
               bounding rectangle is used

            ### ✨ Enhancement Pipeline

            - **2× Upscaling** → Higher resolution for OCR
            - **CLAHE** → Contrast normalization (works in low-light)
            - **Gaussian Denoising** → Smooth out compression artifacts
            - **Adaptive Thresholding** → Clean black-on-white character isolation
            - **Deskewing** → Correct minor rotation using minimum-area rectangle

            """)

        with col2:
            st.markdown("""
            ### ⚙️ Configuration Options

            **Canny Thresholds** (sidebar sliders):
            - Low = 50, High = 150 works for most well-lit images
            - Increase thresholds for noisy/textured backgrounds
            - Decrease for faint/low-contrast plates

            **OCR Languages**:
            - Default: English
            - Add Chinese, Arabic, etc. for multi-language plates
            - More languages = slower inference

            **GPU Acceleration**:
            - Requires NVIDIA CUDA
            - Speeds up EasyOCR 3–5×
            - Not needed for single-image processing

            ---

            ### 📈 OCR Engine Comparison

            | Engine     | Accuracy | Speed  |
            |------------|----------|--------|
            | EasyOCR    | ~70–85%  | Medium |
            | Tesseract  | ~65–75%  | Fast   |
            | PaddleOCR  | ~90–98%  | Slow   |

            ---

            ### 🚀 Extensions & Next Steps

            - **YOLOv8 Detection** – Replace contours with DL detector
            - **Real-time Video** – Process webcam/RTSP streams
            - **Database Logging** – Store plates with timestamps
            - **Number Formatting** – Country-specific plate validation
            - **REST API** – FastAPI/Flask backend for integration

            ---

            ### 📁 Project Structure

            ```
            license_plate_recognition/
            ├── app.py                  ← Streamlit web UI
            ├── lpr_pipeline.py         ← CLI + core pipeline
            ├── requirements.txt        ← Python dependencies
            ├── utils/
            │   ├── preprocessing.py    ← Image preprocessing
            │   ├── detection.py        ← Plate detection
            │   └── recognition.py      ← OCR & post-processing
            ├── sample_images/          ← Test images
            ├── outputs/                ← Saved results
            └── notebooks/
                └── lpr_notebook.ipynb  ← Step-by-step tutorial
            ```
            """)


if __name__ == "__main__":
    main()
