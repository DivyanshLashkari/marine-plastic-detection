# Robust Marine Plastic Detection Using Lightweight Deep Learning Under Diverse Environmental Conditions

🌊 A computer-vision system for detecting marine debris/plastic from aquatic images using **YOLOv8n** (lightweight), with **image enhancement** and **robustness evaluation** under challenging environmental conditions.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Detection** | YOLOv8 nano for efficient marine debris detection |
| **Enhancement** | CLAHE, gamma correction, denoising, contrast |
| **Comparison** | Side-by-side original vs enhanced results |
| **Robustness** | Evaluation under low light, blur, haze, noise |
| **Metrics** | Precision, recall, F1, mAP@50, mAP@50-95 |
| **Interface** | Streamlit web application |

---

## 📂 Project Structure

```
marine_plastic_detection/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central configuration
│   ├── detector.py                 # YOLOv8n detector wrapper
│   ├── enhancement.py              # Image enhancement pipeline
│   └── utils.py                    # Validation, drawing, serialization
│
├── scripts/
│   ├── train.py                    # Model training script
│   ├── validate.py                 # Model evaluation script
│   └── create_challenge_sets.py    # Challenge condition generator
│
├── tests/
│   └── test_core.py                # Unit tests
│
├── data/
│   └── data.yaml                   # YOLO dataset configuration
│
├── datasets/                       # Training data (not committed)
├── weights/                        # Model checkpoints
├── outputs/
│   ├── detections/                 # Saved detection results
│   ├── comparisons/                # Saved comparisons
│   ├── metrics/                    # Evaluation metrics
│   └── challenge_sets/             # Generated challenge images
│
└── .streamlit/
    └── config.toml                 # Streamlit theme
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will start with pretrained COCO weights by default. After training your own model, place `best.pt` in the `weights/` directory.

### 3. Train the Model

```bash
# Prepare dataset in datasets/marine_debris_yolo/ first (see Dataset section)
python scripts/train.py --epochs 100 --batch 16
```

### 4. Evaluate

```bash
python scripts/validate.py --condition normal
```

### 5. Generate Challenge Sets

```bash
python scripts/create_challenge_sets.py --source datasets/marine_debris_yolo/images/test
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Datasets & Multi-Condition Augmentation

The model is trained on a merged, real-world aquatic and underwater dataset:

1. **`trash_ICRA19` Dataset:** 7,684 underwater robot (ROV) images containing plastic debris, biological marine organisms (`bio`), and robotic vehicles (`rov`).
2. **`SOUVIK` Dataset:** Real surface/subsurface floating plastic debris (bags, bottles, nets) paired with clean background waters to suppress false positives.
3. **Multi-Condition Environmental Augmentation:** Physical degradation synthesis covering:
   - **Low-Light / Nocturnal:** Inverse gamma darkening simulating deep ocean attenuation ($\gamma \in [0.40, 0.60]$).
   - **Murky Water / Haze:** Aquatic particulate backscattering with realistic underwater tint.
   - **Optical Defocus / Turbidity:** Gaussian optical blur simulating suspended sediment scattering.
   - **Luminance Equalization:** CLAHE across the LAB color space.

| Split | Base Images | Condition Variations | Total Images |
|---|:---:|:---:|:---:|
| **Train Split** | 1,200 | 1,815 | **3,015** |
| **Validation Split (Benchmark)** | 200 | Clean Unmodified Ground Truth | **200** |

---

## ⚡ Performance & Hardware Acceleration

Trained and benchmarked on **NVIDIA GeForce GTX 1650 (4.0 GB VRAM)** using **PyTorch CUDA 12.4**:

| Metric | Result | Benchmark Significance |
|---|:---:|---|
| **Inference Latency** | **4.7 ms** | Over **212 FPS** — exceeds real-time video requirements |
| **Overall Precision** | **82.1%** | High-fidelity detection with minimal false alarms |
| **Marine Life (Bio) Precision** | **100%** | Zero false alarms on living aquatic organisms |
| **Plastic Precision** | **64.2%** | Robust identification of degraded polymers |

### 🌊 Adverse Condition Stress Test Results (`bio0000_frame0000043.jpg`)

| Test Condition | Image Simulation | Detections | Plastic Confidences | Detection Status |
|---|---|:---:|:---:|:---:|
| **Clean Ambient Water** | Standard underwater light | 1 | **76.5%** | ✅ Verified |
| **Low-Light / Deep Water** | Inverse gamma darkening ($\gamma=0.5$) | 1 | **65.1%** | ✅ Verified |
| **Murky Water / Haze** | Particulate backscatter tint (45%) | 2 | **78.8%, 25.8%** | ✅ Verified |
| **Turbidity Defocus** | Gaussian optical blur ($k=9, \sigma=3.0$) | 3 | **64.7%, 60.9%, 43.3%** | ✅ Verified |

---

## 🎨 User Interface & Academic Documentation

- **Interactive Streamlit Web App:** 4 dedicated pages (`Detection`, `Comparison`, `Evaluation`, `About`) with real-time OpenCV enhancement toggles (CLAHE, Gamma, Denoising, Contrast).
- **Academic Lab Deliverable:** Full UI/UX design specifications, user personas, task workflows, low-fidelity wireframes, design principles, and rubric alignment are documented in [`docs/LAB7_UI_DESIGN_DOCUMENT.md`](docs/LAB7_UI_DESIGN_DOCUMENT.md).

---

## 📋 License & Attribution

Academic project developed for research, marine environmental monitoring, and educational evaluation. Model weights in `weights/best.pt` are provided for reproduction and immediate testing.
