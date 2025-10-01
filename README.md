End-to-End ML Pipeline for Real-Time Material Classification - Scrap Sorting Project

🎯 Project Overview
Hey, this is my take on the scrap material classification pipeline for the AlfaStack assignment. I built a full ML setup that uses computer vision to spot different materials like metal, plastic, paper, glass, cardboard and trash in real time. It's like a simulated conveyor belt that classifies stuff as it goes by. Used TrashNet dataset cuz it's perfect for waste sorting and has good quality images.

📁 Project Structure
├── src/
│   ├── train.py              # the main training script, trains ResNet18 on the data
│   ├── export_model.py       # exports the model to ONNX and TorchScript for faster inference
│   ├── inference.py          # simple script to test single images, outputs class and confidence
│   ├── conveyor_sim.py       # simulates the conveyor belt, processes images one by one with logging
│   └── prepare_data.py       # handles dataset splitting and prep, adapted for the resized TrashNet
├── data/
│   ├── materials/            # the split dataset folders
│   │   ├── train/            # 80% for training
│   │   ├── val/              # 10% for validation
│   │   └── test/             # 10% for testing
│   └── test_images/          # flat folder with sample test images for simulation
├── models/
│   ├── best_model.pth        # the trained PyTorch model checkpoint
│   ├── model.onnx            # ONNX version for lightweight deployment
│   └── model_scripted.pt     # TorchScript version as backup
├── results/
│   ├── metrics.json          # json with accuracy, precision etc
│   ├── confusion_matrix.png  # heatmap of predictions vs actual
│   ├── training_history.png  # plots of loss and acc over epochs
│   └── conveyor_results_*.csv # logs from the simulation runs
├── requirements.txt          # all the pip packages needed
├── README.md                 # this file, explains everything
└── performance_report.md     # detailed report with tables and analysis

🚀 Quick Start
1. Installation
First, make sure you have Python 3.8+ and git.

bash
# Clone or download the repo
git clone <your-repo-url>  # or just unzip if you have the folder
cd thrashnet-project  # or whatever you named it

# Create and activate virtual env (good practice to avoid conflicts)
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate

# Install the deps
pip install -r requirements.txt
# if you have GPU, make sure CUDA PyTorch is installed as per the notes below
2. Dataset Preparation
I used TrashNet dataset, which is great for this cuz it has 6 classes of waste materials. Download it from https://github.com/garythung/trashnet and extract to data/dataset-resized (the resized version is already there in my setup).

Run the prep script to split it:
bash
python src/prepare_data.py
# it auto detects the dataset-resized folder and splits into train/val/test
# no need to choose options, it's set for TrashNet

If you want custom data, edit the main() in prepare_data.py to point to your folder.

3. Train the Model
Now train the model. It uses ResNet18 with transfer learning, should take about 10-20 mins on GPU, longer on CPU.

bash
python src/train.py --device auto --epochs 20 --batch-size 32
# --device auto uses GPU if available, else CPU
# adjust epochs if you want quicker test

It saves the best model to models/best_model.pth and plots to results/.

Training params:
- Architecture: ResNet18 (pretrained on ImageNet, froze early layers)
- Image size: 224x224
- Batch size: 32
- LR: 0.001 with scheduler
- Epochs: 20
- Augmentations: flips, rotations, color changes to handle real world variations

4. Export Model
Export for deployment:
bash
python src/export_model.py
This makes ONNX and TorchScript versions in models/. ONNX is faster for production.

5. Run Inference (Single Image)
Test on one image:
bash
python src/inference.py path/to/your/image.jpg
Outputs the predicted class, confidence, and if it's low confidence.

6. Run Conveyor Simulation
This is the fun part - simulates the conveyor:
bash
# Basic run
python src/conveyor_sim.py --source data/test_images --interval 1.0

# With manual override (for low confidence)
python src/conveyor_sim.py --source data/test_images --override

# Active learning mode (queues misclassifications)
python src/conveyor_sim.py --source data/test_images --active-learning

# Higher threshold
python src/conveyor_sim.py --source data/test_images --threshold 0.8

It processes images one by one, prints results, logs to CSV in results/, flags low confidence (<0.7), and supports override.

Simulation features:
- Real time like, with interval between frames
- Confidence check and warnings
- Logs everything to CSV (timestamp, class, confidence, time taken)
- Manual fix for bad predictions
- Queue for retraining bad ones
- Console shows top 3 guesses with bars

📊 Dataset Information
Dataset: TrashNet (downloaded from github)
Why? It's relevant for scrap, has 6 classes, good quality pics from real waste, about 2500 images total.

Classes and counts (approx):
- Cardboard: 403
- Glass: 501
- Metal: 410
- Paper: 594
- Plastic: 482
- Trash: 137 (bit low, might need more data here)

Alternatives I thought of: TACO dataset, but it's bigger and needs more prep, so stuck with TrashNet for time.

🧠 Model Architecture
ResNet18 with transfer learning - chose it cuz it's fast and accurate for classification, not too heavy for edge.

Why ResNet18?
- Good performance, established for images
- 11.7M params, light for deployment
- Pretrained on ImageNet, good features out of box
- Quick inference, 10-30ms CPU, faster on GPU
- Works great with ONNX for cross platform

Changes I made:
- Froze first 10 layers to speed up training
- Changed last layer for 6 classes
- Fine tuned on waste data

Why not others?
- YOLOv8: too much for just classification, it's for detection
- MobileNet: okay, but ResNet better balance
- Bigger ResNet: overkill, slower

🎯 Performance Metrics
How I evaluated:
- Accuracy overall
- Precision, recall, F1 per class
- Confusion matrix plot
- Expected: 85-92% acc, got around 98% val which is awesome!

From my runs:
- Val Acc: 98.13%
- Inference: 15ms CPU, <5ms GPU
- Model size: 45MB

🚢 Deployment Strategy
Made it lightweight with ONNX:
- Works on anything, fast with ONNX Runtime
- 10-20% quicker than PyTorch on CPU

TorchScript for PyTorch only setups.

Optimizations:
- Batch 1 for real time
- Pre alloc memory
- Quick preprocess

For Jetson Nano/Xavier (bonus):
- Install jetson-stats
- Use TensorRT: convert ONNX to engine for speed
- Run with --device cuda
- Tips: FP16 for 2x speed, quantize for size

🔄 Real-Time Simulation
The conveyor_sim.py mimics belt:
- Processes frames at interval
- Confidence check, low warning
- CSV log all results
- Manual override if wrong
- Active learning queue for bad ones

Outputs:
- conveyor_results_*.csv: all data
- override_log.json: fixes
- retraining_queue.json: for improvement

📈 Performance Report
Check performance_report.md for details:
- Metrics tables
- Class analysis
- Curves
- Benchmarks
- Errors and fixes

🔧 Configuration Options
In train.py edit:
num_epochs = 20  # more if you want
batch_size = 32
lr = 0.001

For inference:
confidence_threshold = 0.7

Simulation args:
--interval 1.0
--threshold 0.7
--override
--active-learning

🐛 Troubleshooting
CUDA memory out:
reduce batch_size to 16 or less in train.py

ONNX not found:
pip install onnxruntime or onnxruntime-gpu for GPU

Deps missing:
pip install -r requirements.txt --upgrade

Dataset issues:
make sure folder structure is right, run prepare_data first, check paths

📝 Development Notes
Tried to keep code modular, separated concerns
Added docs and hints where i could
Handled errors and logs
Followed PEP8 mostly, but not perfect

Future stuff:
- Different thresholds per class
- Video streams
- API for remote
- Dashboard
- Auto retrain
- Versioning models

📚 References
TrashNet: https://github.com/garythung/trashnet
ResNet paper: arxiv.org/abs/1512.03385
ONNX docs: onnx.ai
PyTorch: pytorch.org

👤 Author
Me, for the AlfaStack AI Scrap Sorting assignment. Tested on my Windows laptop with RTX 3060.

📧 Contact
Questions? hiringteampurplecat@gmail.com

📄 License
For assignment only, no commercial use.

Note: This pipeline is pretty much ready for industrial use, modular so you can tweak for your needs. Had fun building it!
