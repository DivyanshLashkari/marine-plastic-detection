"""
Marine Plastic Detection — Streamlit Application

A premium-looking web interface for marine debris / plastic detection
using a lightweight YOLOv8n model, with image enhancement comparison
and robustness evaluation features.

    streamlit run app.py
"""

import io
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_WEIGHTS,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DETECTIONS_DIR,
    COMPARISONS_DIR,
    METRICS_DIR,
)
from src.utils import (
    validate_uploaded_bytes,
    ImageValidationError,
    bgr_to_rgb,
    draw_detections,
    resize_for_display,
    save_detection_result,
    save_comparison_result,
    format_metrics_table,
)
from src.enhancement import (
    enhance_image,
    get_available_methods,
    get_method_description,
)
from src.detector import (
    MarineDebrisDetector,
    DetectorError,
    get_detector,
)


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Marine Plastic Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS — premium dark ocean theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ───────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header ──────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #0e7490 40%, #06b6d4 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.18);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -40%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #f0f9ff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #bae6fd;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── Metric cards ─────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.15);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.3rem 0 0 0;
    }

    /* ── Status badges ────────────────────── */
    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-warning {
        background: rgba(250, 204, 21, 0.15);
        color: #facc15;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-info {
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }

    /* ── Detection table ──────────────────── */
    .det-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 0.5rem;
    }
    .det-table th {
        background: #1e293b;
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.6rem 1rem;
        text-align: left;
        border-bottom: 1px solid #334155;
    }
    .det-table td {
        padding: 0.6rem 1rem;
        color: #e2e8f0;
        border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        font-size: 0.9rem;
    }
    .det-table tr:hover td {
        background: rgba(14, 165, 233, 0.05);
    }

    /* ── Section divider ──────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 2rem 0;
    }

    /* ── Sidebar polish ───────────────────── */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ── Hide Streamlit default elements ──── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_state():
    defaults = {
        "uploaded_image": None,
        "uploaded_name": None,
        "image_bgr": None,
        "detection_result": None,
        "enhanced_image": None,
        "enhanced_result": None,
        "comparison_saved": False,
        "page": "detect",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────
# Model Loading (cached)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_detector():
    """Load and cache the YOLO detector."""
    detector = MarineDebrisDetector()
    detector.load_model()
    return detector


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Marine Plastic Detection</h1>
        <p>Lightweight YOLOv8n · Image Enhancement · Robustness Evaluation</p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        page = st.radio(
            "Select page",
            options=["🔍 Detection", "⚡ Comparison", "📊 Evaluation", "ℹ️ About"],
            label_visibility="collapsed",
        )

        page_map = {
            "🔍 Detection": "detect",
            "⚡ Comparison": "compare",
            "📊 Evaluation": "evaluate",
            "ℹ️ About": "about",
        }
        st.session_state.page = page_map[page]

        st.markdown("---")

        # Model settings
        st.markdown("### 🎯 Detection Settings")
        confidence = st.slider(
            "Confidence Threshold",
            0.05, 1.0,
            CONFIDENCE_THRESHOLD,
            0.05,
            help="Minimum confidence to accept a detection.",
        )
        iou = st.slider(
            "IoU Threshold",
            0.1, 1.0,
            IOU_THRESHOLD,
            0.05,
            help="Intersection over Union threshold for NMS.",
        )

        st.markdown("---")

        # Model info
        try:
            detector = load_detector()
            info = detector.get_model_info()
            st.markdown("### 🤖 Model Info")
            st.markdown(f'<span class="badge-success">✓ Model Loaded</span>', unsafe_allow_html=True)
            st.caption(f"**Model:** {info['model_name']}")
            st.caption(f"**Weights:** `{Path(info['weights']).name}`")
        except Exception:
            st.markdown('<span class="badge-warning">⚠ Model not loaded</span>', unsafe_allow_html=True)

    return confidence, iou


# ──────────────────────────────────────────────
# Sample Images Helper
# ──────────────────────────────────────────────

def get_sample_images():
    """Load curated sample images from the samples/ directory."""
    sample_dir = Path(__file__).resolve().parent / "samples"
    manifest_p = sample_dir / "manifest.json"
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return sample_dir, manifest
        except Exception:
            pass
    return sample_dir, {}


# ──────────────────────────────────────────────
# Page: Detection
# ──────────────────────────────────────────────

def page_detect(confidence: float, iou: float):
    st.markdown("## 🔍 Detect Marine Plastic")
    st.markdown("Upload an aquatic image or choose a curated sample to detect marine debris using YOLOv8n.")

    sample_dir, manifest = get_sample_images()

    col_input_mode, col_settings = st.columns([3, 2])
    with col_input_mode:
        source_mode = st.radio(
            "Select Image Source",
            ["📁 Curated Sample Images", "📤 Upload Custom Image"],
            horizontal=True,
            key="source_mode_detect",
        )

    with col_settings:
        enhancement_method = st.selectbox(
            "Enhancement (optional)",
            ["None"] + [get_method_description(m) for m in get_available_methods()],
            help="Apply an enhancement before detection.",
        )

    img_bgr = None
    image_display_name = ""

    if source_mode == "📁 Curated Sample Images" and manifest:
        sample_options = list(manifest.keys())
        selected_key = st.selectbox(
            "Choose a test sample",
            sample_options,
            format_func=lambda k: f"{manifest[k]} ({k})",
            key="sample_select_detect",
        )
        sample_path = sample_dir / selected_key
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            image_display_name = manifest.get(selected_key, selected_key)
            st.session_state.image_bgr = img_bgr
            st.session_state.uploaded_name = selected_key
    else:
        uploaded = st.file_uploader(
            "Upload an aquatic image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            help="Supported formats: JPG, PNG, BMP, TIFF",
            key="file_uploader_detect",
        )
        if uploaded is not None:
            try:
                file_bytes = uploaded.read()
                img_bgr = validate_uploaded_bytes(file_bytes, uploaded.name)
                image_display_name = uploaded.name
                st.session_state.image_bgr = img_bgr
                st.session_state.uploaded_name = uploaded.name
            except ImageValidationError as e:
                st.error(f"❌ {e}")
                return

    if img_bgr is not None:
        # Preview
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        col_preview, col_result = st.columns(2)

        with col_preview:
            st.markdown("#### 📷 Input Image")
            display_img = resize_for_display(bgr_to_rgb(img_bgr))
            st.image(display_img, use_container_width=True)
            h, w = img_bgr.shape[:2]
            st.caption(f"{image_display_name}  ·  {w}×{h} px")

        # Detect
        if st.button("🚀  Run Detection", use_container_width=True, type="primary"):
            with st.spinner("Running detection…"):
                try:
                    detector = load_detector()
                    process_img = img_bgr.copy()
                    applied_enhancement = None

                    if enhancement_method != "None":
                        method_key = enhancement_method.split("—")[0].strip().lower()
                        method_map = {get_method_description(m).split("—")[0].strip().lower(): m
                                      for m in get_available_methods()}
                        if method_key in method_map:
                            applied_enhancement = method_map[method_key]
                            process_img = enhance_image(img_bgr, applied_enhancement)

                    result = detector.detect(
                        process_img,
                        confidence=confidence,
                        iou=iou,
                    )

                    st.session_state.detection_result = result
                    st.session_state.enhanced_image = process_img if applied_enhancement else None

                except DetectorError as e:
                    st.error(f"❌ Detection failed: {e}")
                    return
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    return

        # Show results
        result = st.session_state.detection_result
        if result is not None:
            detections = result["detections"]
            process_img = st.session_state.enhanced_image if st.session_state.enhanced_image is not None else img_bgr

            with col_result:
                st.markdown("#### 🎯 Detection Result")
                annotated = draw_detections(process_img, detections)
                st.image(bgr_to_rgb(resize_for_display(annotated)), use_container_width=True)

            # Metrics row
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="value">{result['num_detections']}</p>
                    <p class="label">Total Detections</p>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
                st.markdown(f"""
                <div class="metric-card">
                    <p class="value">{avg_conf:.2f}</p>
                    <p class="label">Avg Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                max_conf = max((d["confidence"] for d in detections), default=0)
                st.markdown(f"""
                <div class="metric-card">
                    <p class="value">{max_conf:.2f}</p>
                    <p class="label">Max Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="value">{result['inference_time_ms']:.0f}ms</p>
                    <p class="label">Inference Time</p>
                </div>
                """, unsafe_allow_html=True)

            # Detection table
            if detections:
                st.markdown("#### 📋 Detection Details")
                rows = []
                for i, d in enumerate(detections, 1):
                    bbox = d["bbox"]
                    rows.append({
                        "#": i,
                        "Class": d["class_name"].upper(),
                        "Confidence": f"{d['confidence']:.4f}",
                        "BBox (x1,y1,x2,y2)": f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Save result
                if st.button("💾 Save Result", key="save_det"):
                    model_info = detector.get_model_info()
                    path = save_detection_result(
                        st.session_state.uploaded_name or "sample.jpg",
                        detections, model_info,
                        enhancement=None,
                        inference_time_ms=result["inference_time_ms"],
                    )
                    st.success(f"Result saved → `{path.name}`")
            else:
                st.info("ℹ️ No marine debris detected above the confidence threshold.")


# ──────────────────────────────────────────────
# Page: Comparison
# ──────────────────────────────────────────────

def page_compare(confidence: float, iou: float):
    st.markdown("## ⚡ Original vs Enhanced Comparison")
    st.markdown("Compare detection results between original and enhanced images side by side.")

    sample_dir, manifest = get_sample_images()

    col_input, col_enh = st.columns([3, 2])
    with col_input:
        source_mode = st.radio(
            "Select Image Source",
            ["📁 Curated Sample Images", "📤 Upload Custom Image"],
            horizontal=True,
            key="source_mode_compare",
        )

    with col_enh:
        method = st.selectbox(
            "Enhancement method",
            get_available_methods(),
            format_func=get_method_description,
        )

    img_bgr = None
    if source_mode == "📁 Curated Sample Images" and manifest:
        sample_options = list(manifest.keys())
        selected_key = st.selectbox(
            "Choose a test sample",
            sample_options,
            format_func=lambda k: f"{manifest[k]} ({k})",
            key="sample_select_compare",
        )
        sample_path = sample_dir / selected_key
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
    else:
        uploaded = st.file_uploader(
            "Upload an image for comparison",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="file_uploader_compare",
        )
        if uploaded is not None:
            try:
                file_bytes = uploaded.read()
                img_bgr = validate_uploaded_bytes(file_bytes, uploaded.name)
            except ImageValidationError as e:
                st.error(f"❌ {e}")
                return

    if img_bgr is not None:
        if st.button("🔄  Run Comparison", use_container_width=True, type="primary"):
            with st.spinner("Processing original and enhanced images…"):
                detector = load_detector()

                # Original
                orig_result = detector.detect(img_bgr, confidence=confidence, iou=iou)

                # Enhanced
                enhanced_bgr = enhance_image(img_bgr, method)
                enh_result = detector.detect(enhanced_bgr, confidence=confidence, iou=iou)

            # Side-by-side
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            col_o, col_e = st.columns(2)

            with col_o:
                st.markdown("#### 📷 Original")
                annotated_o = draw_detections(img_bgr, orig_result["detections"])
                st.image(bgr_to_rgb(resize_for_display(annotated_o)), use_container_width=True)

                st.markdown(f"""
                <div class="metric-card">
                    <p class="value">{orig_result['num_detections']}</p>
                    <p class="label">Detections</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"⏱ {orig_result['inference_time_ms']:.0f} ms")

            with col_e:
                st.markdown(f"#### ✨ Enhanced ({get_method_description(method).split('—')[0].strip()})")
                annotated_e = draw_detections(enhanced_bgr, enh_result["detections"])
                st.image(bgr_to_rgb(resize_for_display(annotated_e)), use_container_width=True)

                st.markdown(f"""
                <div class="metric-card">
                    <p class="value">{enh_result['num_detections']}</p>
                    <p class="label">Detections</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"⏱ {enh_result['inference_time_ms']:.0f} ms")

            # Comparison summary
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### 📊 Comparison Summary")

            delta_det = enh_result["num_detections"] - orig_result["num_detections"]
            avg_o = np.mean([d["confidence"] for d in orig_result["detections"]]) if orig_result["detections"] else 0
            avg_e = np.mean([d["confidence"] for d in enh_result["detections"]]) if enh_result["detections"] else 0

            comp_df = pd.DataFrame({
                "Metric": ["Detections", "Avg Confidence", "Inference (ms)"],
                "Original": [
                    str(orig_result["num_detections"]),
                    f"{avg_o:.4f}",
                    f"{orig_result['inference_time_ms']:.1f}",
                ],
                "Enhanced": [
                    str(enh_result["num_detections"]),
                    f"{avg_e:.4f}",
                    f"{enh_result['inference_time_ms']:.1f}",
                ],
                "Delta": [
                    f"{delta_det:+d}",
                    f"{avg_e - avg_o:+.4f}",
                    f"{enh_result['inference_time_ms'] - orig_result['inference_time_ms']:+.1f}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)


            # Save
            if st.button("💾 Save Comparison", key="save_cmp"):
                model_info = detector.get_model_info()
                path = save_comparison_result(
                    {"detections": orig_result["detections"], "num": orig_result["num_detections"]},
                    {"detections": enh_result["detections"], "num": enh_result["num_detections"]},
                    method,
                )
                st.success(f"Comparison saved → `{path.name}`")


# ──────────────────────────────────────────────
# Page: Evaluation
# ──────────────────────────────────────────────

def page_evaluate():
    st.markdown("## 📊 Evaluation Results")
    st.markdown("View saved evaluation metrics from robustness experiments.")

    # Load saved metrics
    metric_files = sorted(METRICS_DIR.glob("*.json"))

    if not metric_files:
        st.info(
            "ℹ️ No evaluation results found yet.\n\n"
            "Run the validation script to generate metrics:\n"
            "```\npython scripts/validate.py --condition normal\n```"
        )
        return

    records = []
    for f in metric_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        row = {"Condition": data.get("condition", "—")}
        row.update(data.get("metrics", {}))
        row["Timestamp"] = data.get("timestamp", "")
        records.append(row)

    df = pd.DataFrame(records)


    # Summary table
    st.dataframe(
        df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "mAP50": "{:.4f}",
            "mAP50_95": "{:.4f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    # Chart
    if len(df) > 1:
        st.markdown("#### Performance Across Conditions")
        chart_metrics = ["precision", "recall", "f1_score", "mAP50"]
        available = [m for m in chart_metrics if m in df.columns]

        if available:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#1e293b")

            x = np.arange(len(df))
            width = 0.18
            colors = ["#38bdf8", "#4ade80", "#facc15", "#f472b6"]

            for i, metric in enumerate(available):
                vals = pd.to_numeric(df[metric], errors="coerce").fillna(0)
                bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)

            ax.set_xticks(x + width * (len(available) - 1) / 2)
            ax.set_xticklabels(df["Condition"], color="#94a3b8", fontsize=10)
            ax.set_ylabel("Score", color="#94a3b8")
            ax.set_ylim(0, 1.05)
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
            ax.tick_params(colors="#64748b")
            ax.spines[:].set_color("#334155")

            st.pyplot(fig)


# ──────────────────────────────────────────────
# Page: About
# ──────────────────────────────────────────────

def page_about():
    st.markdown("## ℹ️ About This Project")

    st.markdown("""
    ### Robust Marine Plastic Detection Using Lightweight Deep Learning Under Diverse Environmental Conditions

    This application implements a **lightweight YOLOv8n-based detection system**
    for identifying marine debris / plastic in aquatic imagery.

    #### 🎯 Key Features

    | Feature | Description |
    |---------|-------------|
    | **Detection** | YOLOv8 nano for efficient marine debris detection |
    | **Enhancement** | CLAHE, gamma correction, denoising, contrast adjustment |
    | **Comparison** | Side-by-side original vs enhanced detection results |
    | **Robustness** | Evaluation under low light, blur, haze, and noise |
    | **Metrics** | Precision, recall, F1-score, mAP@50, mAP@50-95 |

    #### 📂 Tech Stack

    - **Model:** YOLOv8n (Ultralytics) — lightweight object detection
    - **Framework:** PyTorch
    - **CV Library:** OpenCV
    - **Interface:** Streamlit
    - **Dataset:** NASA Marine Debris (PlanetScope imagery)

    #### ⚠️ Important Note

    > The underlying dataset uses a **`marine_debris`** class label.
    > While the project is framed around plastic detection, the ground-truth
    > annotations encompass various debris types and cannot be assumed to
    > represent exclusively plastic materials. This is documented as a
    > known limitation.

    #### 📄 Base Paper

    *"Marine Debris Detection in Satellite Surveillance Using Attention
    Mechanisms"* — IEEE JSTARS, 2024.

    Our project differentiates by focusing on **robustness under diverse
    environmental conditions** with **image enhancement** and **lightweight
    efficient detection**, rather than attention mechanisms.
    """)


# ──────────────────────────────────────────────
# Main Router
# ──────────────────────────────────────────────

def main():
    render_header()
    confidence, iou = render_sidebar()

    page = st.session_state.page

    if page == "detect":
        page_detect(confidence, iou)
    elif page == "compare":
        page_compare(confidence, iou)
    elif page == "evaluate":
        page_evaluate()
    elif page == "about":
        page_about()


if __name__ == "__main__":
    main()
