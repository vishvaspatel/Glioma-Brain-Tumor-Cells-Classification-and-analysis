# Glioma Brain‑Tumor Cell Classification

> **Timeline:** **Aug 2024 – Nov 2024**   |   **Collaborators:** IIT Jodhpur × AIIMS Jodhpur   |   **Live demo:** [https://glioma‑cells‑demo.streamlit.app](https://glioma‑cells‑demo.streamlit.app)   |   **Repo:** [https://github.com/YOUR‑ORG/glioma‑cell‑classify](https://github.com/YOUR‑ORG/glioma‑cell‑classify)

---

## 1  Project at a Glance

Automatically locating and classifying the cellular landscape of glioma biopsy images accelerates diagnosis, treatment planning, and therapeutic research. We benchmarked three deep‑learning pipelines built around **YOLOv8** detection and multi‑head classifiers to identify *astrocytes*, *microglia*, and *cancerous glioma* cells.

| Pipeline             | Detector               | Classifier                  | Cell‑Type Acc. | Binary (Cancer vs Normal) |
| -------------------- | ---------------------- | --------------------------- | -------------- | ------------------------- |
| 1. **Single‑Stage**  | YOLOv8‑Seg (3 classes) | —                           | 40 %           | —                         |
| 2. **Two‑Stage CNN** | YOLOv8                 | ResNet‑based Multi‑Head CNN | 64 %           | 93 %\*                    |
| 3. **Two‑Stage ViT** | YOLOv8                 | ViT‑based Multi‑Head        | **94 %**       | **98 %**\*                |

---

## 2  Dataset

The primary dataset consists of bright‑field microscopic tiles annotated by pathologists at AIIMS Jodhpur.

| Class            | Images |
| ---------------- | ------ |
| Astrocyte        | 181    |
| Microglia        | 56     |
| Cancerous Glioma | 201    |

All images were resized to **512 × 512** px, normalized to ImageNet statistics, and augmented with flip, affine, and color‑jitter before training.

---

## 3  Methodology

### 3.1  Single‑Stage YOLOv8 Segmentation (Baseline)

* **Goal:** Direct instance segmentation & classification in one network.
* **Loss:** Composite (box + cls + mask).
* **Limitation:** Struggles to separate overlapping microglia; low F1 for minority class.

### 3.2  YOLOv8 + Multi‑Head CNN

1. **Detection:** YOLOv8 (bounding boxes only).
2. **Cropping:** Detected crops (128 × 128) passed to a ResNet‑50 backbone.
3. **Heads:**

   * **Head‑A (3‑way):** Astrocyte / Microglia / Cancerous.
   * **Head‑B (2‑way):** Cancerous / Normal.
4. **Loss:** `CE_HA + λ · BCE_HB`, λ = 0.3.

### 3.3  YOLOv8 + Multi‑Head Vision Transformer *(Best)*

* Replaces ResNet with a ViT pretrained.
* Multi‑head outputs identical to §3.2.
* Large receptive field → superior context capture → +30 pp accuracy.
---

## 4  Quick Start

```bash
# 1. Clone
$ git clone https://github.com/YOUR-ORG/glioma-cell-classify.git
$ cd glioma-cell-classify

# 2. Create env
$ conda create -n glioma python=3.10 && conda activate glioma
$ pip install -r requirements.txt  # ultralytics, timm, torch, streamlit …

# 3. Inference demo
$ python infer.py --weights weights/best_vit.pt --source data/sample.png

# 4. Streamlit app
$ streamlit run app.py
```

---

## 5  Repository Layout

```
├── app.py                # Streamlit UI
├── configs/              # YAML training configs
├── data/                 # ↳ images.zip downloaded by script
├── notebooks/            # EDA & prototyping
├── src/
│   ├── detectors.py      # YOLO wrapper
│   ├── classifiers.py    # CNN & ViT heads
│   └── train.py
└── reports/
    └── metrics.xlsx      # Full results & confusion matrices
```

---

## 6  Tech Stack

* **Detection:** [YOLOv8](https://github.com/ultralytics/ultralytics)
* **Classification:** PyTorch ▸ ResNet‑50 ▸ ViT‑B/16 (timm)
* **Web Demo:** Streamlit + OpenCV

---

## 7  How to Cite

```bibtex
@unpublished{sohail2024glioma,
  title     = {AI-Driven Approach for Identifying Astrocytes, Microglia, and Cancerous Glioma Cells},
  author    = {Sohail, Md. Aamir and Patel, Vishvas and Patel, Om and Maurya, Rahul},
  year      = {2024},
  note      = {Joint project, IIT Jodhpur & AIIMS Jodhpur}
}
```

---

## 8  License

This work is licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

## 9  Acknowledgements

Special thanks to **AIIMS Jodhpur Pathology Dept.** for expert annotations and to **HPC‑IITJ** for GPU compute.

---

<p align="center"><sub>Made with ♥ at IIT Jodhpur.</sub></p>
