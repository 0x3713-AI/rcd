# Papers & Datasets

A reading list and data catalogue for this project: an open-ended
exploration of deep-learning-based road crack detection for UAV imagery,
starting from the URCD-YOLO paper but explicitly reaching beyond it.

`papers and datasets mentioned here are just suggestions, and could be not that promising or the best choice to spent time reading`
---

### URCD-YOLO — Deep learning-based road crack detection for UAV imagery

- **Authors:** Weiguo Yi, Longteng Wang, Lingwei Yan
- **Venue:** *Egyptian Informatics Journal*, vol. 34, art. 100985, 2026
- **DOI:** [10.1016/j.eij.2026.100985](https://doi.org/10.1016/j.eij.2026.100985)

**Problem.** UAV aerial imagery road crack detection faces three core
bottlenecks: (1) high miss rate of tiny cracks, (2) insufficient
fine-grained localization accuracy, and (3) excessive parameters hindering
UAV/edge deployment.

**Approach.** URCD-YOLO builds on **YOLO11s** and makes four targeted
optimizations balancing detection accuracy against a lightweight design:

1. **Improved BiFPN** in the neck — replaces conventional concatenation with
   **Softmax-normalized weighted fusion** for bidirectional feature fusion,
   reducing tiny-crack feature loss and fusion deviation during multi-scale
   aggregation.
2. **WADown (Weighted Adaptive Dual-path Downsampling)** — replaces vanilla
   convolutional downsampling in backbone and neck, cutting parameter
   overhead while preserving micro-crack edge and texture detail.
3. **WTConv (Wavelet Transform Convolution)** embedded into the **C3k2**
   module — enhances perception of low-frequency global road features and
   fine-grained linear crack textures.
4. **MSCA (Multi-scale Channel Attention)** — a cross-domain collaborative
   optimization framework spanning spatial, channel, and frequency
   dimensions, boosting feature capture for small cracks in complex
   backgrounds.

**Results.** On the public **UAV-PDD2023** dataset, URCD-YOLO achieves
**+9.1%** `mAP@0.5` and **−9.5%** parameters vs. the YOLO11s baseline,
motivating it as a lightweight high-precision solution for UAV and other
resource-constrained edge terminals.

---

## Prerequisite / foundational papers

These are the building blocks the anchor paper composes, if time allows read them, or skim, to understand the field of the problem.

### Detection backbones & frameworks

- **YOLO11: An Overview of the Key Architectural Enhancements** — Ultralytics,
  arXiv:2410.17725, 2024. The baseline model (YOLO11s); introduces the
  lightweight C3k2 module and decoupled heads that URCD-YOLO modifies.
- **YOLOv5/YOLOv8 lineages** — earlier iterations of the same family; most
  UAV crack-detection literature builds on YOLOv5s/YOLOv8s, so reading these
  contextualizes where YOLO11 stands.

### Feature pyramids & fusion (motivates the improved BiFPN)

- **Feature Pyramid Networks for Object Detection** — Lin et al., CVPR 2017.
  The original multi-scale feature pyramid.
- **Path Aggregation Network for Instance Segmentation** — Liu et al., CVPR
  2018 (PANet). Adds bottom-up path aggregation.
- **EfficientDet: Scalable and Efficient Object Detection** — Tan, Pang & Le,
  CVPR 2020. Introduces **BiFPN** (bidirectional feature pyramid network with
  learnable weighted fusion) — the direct ancestor of URCD-YOLO's improved
  BiFPN. Read this to understand what the Softmax-weighted variant changes.

### Frequency / large-receptive-field convolution (motivates WTConv)

- **Wavelet Convolutions for Large Receptive Fields** — Finder et al., ECCV
  2024, arXiv:2407.05848. The **WTConv** layer itself: cascaded wavelet
  decomposition gives a large effective receptive field with logarithmic
  parameter growth and better low-frequency response. This is the technique
  URCD-YOLO embeds into C3k2.

### Attention mechanisms (motivates MSCA)

- **Squeeze-and-Excitation Networks** — Hu, Shen & Sun, CVPR 2018. Channel
  attention foundation.
- **CBAM: Convolutional Block Attention Module** — Woo et al., ECCV 2018.
  Channel + spatial attention; the common baseline that multi-scale channel
  attention extends.
- **Multi-Scale Channel Attention (MSCA)** — note the name is shared; the
  anchor paper's MSCA is its own design, but **SegNeXt: Rethinking
  Convolutional Attention Design for Semantic Segmentation** (Guo et al.,
  NeurIPS 2022) also uses an MSCA-like multi-scale channel attention block
  and is worth reading for background.

### Lightweight / efficient design (motivates WADown & the "−9.5% params" goal)

- **GhostNet: More Features from Cheap Operations** — Han et al., CVPR 2020.
  Parameter-efficient feature generation; representative of the
  "cheap operation replaces heavy convolution" idea.
- **MobileNets** (V1/V2/V3) — the canonical lightweight conv architectures;
  good baseline knowledge for edge deployment claims.

---

## Adjacent / closely-related works (UAV road crack & defect detection)

Recent work on the same dataset and task, useful as comparison baselines and
for spotting where the field is heading:

- **AC-YOLO** — improved YOLOv8s for UAV aerial crack detection using
  LSK-attention + BiFPN + WIoUv3 (Computer Engineering and Applications,
  2025). Comparable stack to URCD-YOLO, a useful ablation-companion.
- **YOLOv8s-LS** — lightweight UAV crack detector with ghost conv, dilated
  local attention (C2f_MDLA), MNSA, and separable conv; reports strong
  mAP50 gains on UAV-PDD2023 (IOP, 2025).
- **YOLOv8-LUAPD** — lightweight YOLOv8n variant with multi-kernel conv
  shuffle + self-calibrated local channel attention + WIoUv3, deployed on
  RK3588 at 33 FPS (IOP, 2025). Good edge-deployment reference point.
- **RDD-YOLOv5** — YOLOv5 with Swin-Transformer + EVC block for UAV road
  defect detection (Sensors, 2023). A transformer-flavored counterpoint.
- **RPP-YOLOv11** — YOLOv11 with RGB-T multispectral fusion, windmill-shaped
  PSConv, and a P4–P6 detection pyramid (PLOS ONE, 2026). Demonstrates the
  four-scale / multimodal direction.
- **RC-DETR** — a real-time DETR-based detector (RT-DETR baseline) with
  receptive-field context aggregation and learnable B-spline (KAN-style)
  basis functions; reports 85.5% mAP50 on UAV-PDD2023 (IOP, 2025). Directly
  relevant to the transformer path we want to explore.
- **URD-YOLOv11n** — YOLOv11n with SCSA attention, DySample upsampling, LSK
  attention, and Wasserstein distance loss for low-resolution UAV defects
  (Scientific Reports, 2026).
- **EFA-Net** — YOLO12n-based edge & frequency aware network for pavement
  distress (Journal of Computer Applications, 2026). Another frequency-aware
  design, useful for comparing the WTConv direction.
- **YOLO11-MBC** — YOLO11 with multi-scale feature-fusion backbone +
  BiFPN + multimodal cross-attention for road cracks on RDD2022 (MDPI
  Sensors, 2025).

---

## Datasets

### UAV-PDD2023 (primary benchmark)

The dataset the anchor paper evaluates on.

- **Description:** 2,439–2,440 UAV images, 11,150+ instances, captured at
  ~30 m altitude in China across two weather conditions and varying road
  construction quality (highways, provincial, county roads).
- **Classes (6):** Longitudinal Crack (LC), Transverse Crack (TC), Alligator
  Crack (AC), Oblique Crack (OC), Repair (RP), Pothole (PH).
- **Format:** PASCAL VOC; also mirrored on Hugging Face as
  [`vikhyatk/uav-pdd2023`](https://huggingface.co/datasets/vikhyatk/uav-pdd2023)
  (used by `notebooks/data_prep.py`).
- **Paper:** "UAV-PDD2023: A benchmark dataset for pavement distress
  detection based on UAV images", *Data in Brief*, 2023,
  DOI [10.1016/j.dib.2023.109692](https://doi.org/10.1016/j.dib.2023.109692).
- **Data:** DOI [10.5281/zenodo.8214118](https://zenodo.org/records/8214118)
  (or 8429208).

### RDD-2022 (multi-national, ground-level)

The standard large-scale road-damage dataset; the primary component of the
Unified dataset below.

- **Description:** 47,420 images, 55,000+ instances, from Japan, India,
  Czech Republic, Norway, United States, and China.
- **Classes (relevant 4):** D00 Longitudinal, D10 Transverse, D20 Alligator,
  D40 Pothole.
- **Paper:** "RDD2022: A multi-national image dataset for automatic road
  damage detection", *Geoscience Data Journal*, 2022,
  DOI [10.1002/gdj3.260](https://doi.org/10.1002/gdj3.260);
  arXiv:2209.08538.
- **Tooling:** sekilab/RoadDamageDetector on GitHub.

### Unified Road Defect Dataset (larger, merged — recommended next step)

A merged YOLO-format dataset that combines RDD-2022 (ground-level) with two
supplementary aerial/drone sets, UAV-PDD2023 (China) and RoadDamageVision
(China + Spain), into a single 4-class **CRDDC** schema.

- **Hugging Face:** [`TamAko783/Unified_Road_Defect_Dataset`](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset)
- **Size:** ~30k images / ~75.7k instances (train ~59.2k rows incl.
  hard-negatives, val 9.02k); images 512–4040 px.
- **Classes (4):** D00 Longitudinal, D10 Transverse, D20 Alligator, D40
  Pothole.
- **Format:** Ultralytics YOLO layout (`images/`, `labels/`, `.yaml`),
  source-prefixed filenames (`rdd_`, `uav_`, `rdv_`) so per-source metrics
  are reportable.
- **Caveat:** this is a **derived dataset**, cite all three source datasets
  (RDD-2022, UAV-PDD2023, RoadDamageVision) and verify each source's license
  before redistribution. The merged schema drops oblique cracks and repair
  classes, so comparisons against the 6-class UAV-PDD2023 must be mapped
  carefully.

### RoadDamageVision (supplementary aerial)

- **Description:** drone-captured road-surface images from China + Spain,
  manually annotated (6 classes: D00/D10/D20/D40, Repair, Block Crack);
  7,647 instances, COCO format; D40 (pothole) most common.
- **Source:** Mendeley Data `ypm4h4z25c` v3,
  DOI [10.17632/ypm4h4z25c.3](https://doi.org/10.17632/ypm4h4z25c.3)
  (CC BY 4.0).
- **Relevance:** another aerial/drone source for cross-dataset
  generalization testing.

---

## Suggested order

1. FPN → PANet → EfficientDet (BiFPN) (how multi-scale fusion evolved)
2. YOLO11 overview (the baseline)
3. WTConv (ECCV 2024) (the frequency-domain building block)
4. The anchor paper (URCD-YOLO)
5. RC-DETR / transformer route, or the newer YOLO variants
   (YOLOv8s-LS, YOLOv11-MBC, EFA-Net) for comparison, and the Unified Road
   Defect Dataset for scale-up.
