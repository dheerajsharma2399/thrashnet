# End-to-End ML Pipeline for Real-Time Scrap Material Classification

## Project Overview
This project implements a computer vision-based system for classifying scrap materials (cardboard, glass, metal, paper, plastic, trash) in real-time, simulating a conveyor belt sorting process. Built for the AlfaStack assignment, it leverages the TrashNet dataset and a ResNet18 model for efficient inference, suitable for industrial deployment on edge devices like NVIDIA Jetson.

## Dataset Used & Why
The primary dataset is TrashNet, sourced from [GitHub](https://github.com/garythung/trashnet), containing ~2,527 real-world images across 6 classes of waste materials. Images are resized to 224x224 for model compatibility.

**Class Distribution (Approximate):**
- Cardboard: 403 images (15.9%)
- Glass: 501 (19.8%)
- Metal: 410 (16.2%)
- Paper: 594 (23.5%)
- Plastic: 482 (19.1%)
- Trash: 137 (5.4%) – underrepresented, addressed via augmentations

**Why TrashNet?** It's directly relevant to scrap sorting with high-quality, diverse images capturing natural variations in lighting, angles, and textures. Alternatives like TACO were considered but dismissed due to larger size and preprocessing overhead; TrashNet enables quick prototyping while maintaining realism for conveyor simulations. The dataset is split 80/10/10 (train/val/test) using [src/prepare_data.py](src/prepare_data.py), with augmentations to mitigate imbalance.

## Architecture & Training Process
The core model is ResNet18 (pretrained on ImageNet), selected for its balance of accuracy (top-1 ~69% on ImageNet) and efficiency (11.7M parameters, ~45MB size). Early layers are frozen to leverage transfer learning, reducing training time while fine-tuning the classifier head for 6 classes.

**Key Architectural Changes:**
- Input: 224x224x3 RGB images
- Backbone: ResNet18 (frozen first ~10 layers)
- Head: Global average pooling + FC layer (512 → 6) + Softmax
- Output: Class probabilities

**Training Process:**
- **Framework:** PyTorch 2.0 with TorchVision
- **Optimizer:** Adam (LR=0.001, weight decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Loss:** CrossEntropyLoss (no class weights initially; future: weighted for imbalance)
- **Data Augmentations (Training):** Random horizontal flip (p=0.5), rotation (±15°), color jitter (±0.2), affine translation (±10%), ImageNet normalization
- **Validation/Test:** Resize + normalization only
- **Hyperparameters:** Batch size=32, epochs=20, early stopping via best val_acc
- **Hardware:** GPU (CUDA) if available; ~10-20 mins on RTX 3060
- **Evaluation:** Accuracy, precision/recall/F1 (weighted), confusion matrix

Training script [src/train.py](src/train.py) saves the best checkpoint to models/best_model.pth and generates plots/metrics in results/. Achieved ~98% validation accuracy in runs, with steady convergence and minimal overfitting.

## Deployment Decisions
For production, the model is exported to ONNX (primary) and TorchScript (fallback) via [src/export_model.py](src/export_model.py), enabling cross-platform inference with ONNX Runtime (~10-20% faster than PyTorch on CPU).

**Rationale:**
- **Lightweight:** ONNX reduces dependencies; supports quantization (INT8/FP16) for edge devices (e.g., Jetson Nano: ~25ms inference post-TensorRT)
- **Real-Time:** Batch=1 inference <30ms CPU / <5ms GPU, suitable for 1-2 FPS conveyor simulation
- **Optimizations:** Pre-allocated memory, minimal preprocessing; future: TensorRT for NVIDIA hardware
- **Fallbacks:** TorchScript for PyTorch ecosystems; no cloud dependency for local deployment
- **GUI Integration:** Tkinter-based app in [src/gui_app.py](src/gui_app.py) for interactive testing; conveyor simulation in [src/conveyor_sim.py](src/conveyor_sim.py) mimics industrial flow with logging/overrides

Avoided heavier frameworks (e.g., YOLO for detection) as classification suffices; prioritized modularity for scalability (multi-instance via load balancing).

## Folder Structure
```
.
├── src/                          # Source code
│   ├── prepare_data.py           # Dataset splitting and prep
│   ├── train.py                  # Model training
│   ├── export_model.py           # Export to ONNX/TorchScript
│   ├── inference.py              # Single/batch prediction
│   ├── conveyor_sim.py           # Real-time simulation
│   ├── gui_app.py                # GUI for interactive classification
│   └── gui_test_app.py           # Test GUI variant
├── data/                         # Dataset (not committed; download separately)
│   └── materials/                # Split folders: train/val/test per class
├── models/                       # Saved models
│   ├── best_model.pth            # Trained checkpoint
│   ├── model.onnx                # ONNX export
│   └── model_scripted.pt         # TorchScript export
├── results/                      # Outputs and logs
│   ├── metrics.json              # Evaluation metrics
│   ├── confusion_matrix.png      # Prediction heatmap
│   ├── training_history.png      # Loss/acc curves
│   └── conveyor_results_*.csv    # Simulation logs
├── requirements.txt              # Dependencies
├── setup.sh                      # Environment setup
├── run_pipeline.py               # End-to-end runner
├── README.md                     # This file
└── performance_report.md         # Performance summary
```

## How to Run
### 1. Setup Environment
```bash
# Clone/download project
git clone <repo-url>  # or extract folder
cd thrashnet

# Run setup (creates venv, installs deps)
bash setup.sh  # On Windows: Use Git Bash or adapt to .bat

# Or manually:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# For GPU: Ensure CUDA-compatible PyTorch (see requirements.txt notes)
```

### 2. Prepare Dataset
Download TrashNet to `data/dataset-resized` (manual; ~100MB). Run:
```bash
python src/prepare_data.py
```
This splits into `data/materials/train|val|test`.

### 3. Train Model
```bash
python src/train.py --epochs 20 --batch-size 32 --device auto
```
Outputs: models/best_model.pth, results/{metrics.json, plots}.

### 4. Export Model
```bash
python src/export_model.py
```
Generates ONNX/TorchScript in models/.

### 5. Run Inference/Simulation
- **Single Image:**
  ```bash
  python src/inference.py path/to/image.jpg
  ```
- **Conveyor Simulation:**
  ```bash
  python src/conveyor_sim.py --source data/materials/test --interval 1.0 --threshold 0.7
  ```
  Flags low-confidence (<0.7); supports --override and --active-learning.

### 6. Run the App (GUI)
For interactive classification:
```bash
python src/gui_app.py
```
- Load model (auto-detects ONNX/PyTorch)
- Upload image or use webcam
- View prediction, confidence, and top-3 classes
- Test mode via [src/gui_test_app.py](src/gui_test_app.py) for batch processing

### 7. End-to-End Pipeline
```bash
python run_pipeline.py  # Trains, exports, runs simulation + GUI demo
```

**Troubleshooting:**
- CUDA OOM: Reduce batch-size to 16.
- Missing deps: `pip install -r requirements.txt --upgrade`.
- Dataset errors: Verify `data/materials` structure post-prep.

For full performance details, see [performance_report.md](performance_report.md). This pipeline is modular and ready for extension (e.g., API via Flask).

**Author:** Personal project for scrap classification. Contact: dheerajoffical2306@gmail.com  
**License:** Assignment use only.
