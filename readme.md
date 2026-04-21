# VadCLIP-multi
Official PyTorch implementation of the manuscript "Multi-Scale Temporal Feature Fusion with Cross-Scale Alignment for Weakly Supervised Video Anomaly Detection", submitted to The NeuroComputing.

## Project Overview
VadCLIP-multi is a multi-scale learning project based on the CLIP (Contrastive Language-Image Pre-training) model, focusing on video anomaly detection and cross-modal understanding tasks. The system implements multi-scale feature extraction and segmentation-based approaches for improved anomaly detection in videos.

## Features
- Multi-scale feature extraction for comprehensive video analysis
- Segmentation-based anomaly detection using CLIP architecture
- Video anomaly detection with cross-modal understanding
- Support for UCF-Crime and XD-Violence datasets

## Project Structure
src/
├── clip/                 # CLIP model implementation
│   ├── __init__.py
│   ├── clip.py
│   ├── model.py
│   └── simple_tokenizer.py
├── utils/                # Utility functions
│   ├── dataset.py
│   ├── dataset_seg.py
│   ├── layers.py
│   ├── lr_warmup.py
│   ├── tools.py
│   ├── ucf_detectionMAP.py
│   └── xd_detectionMAP.py
├── model_seg.py          # Multi-scale segmentation model
├── ucf_*.py              # UCF-Crime dataset training/testing scripts
├── xd_*.py               # XD-Violence dataset training/testing scripts
└── crop.py               # Image/video cropping utilities

### Configuration
Dataset-specific options available in ucf_option_seg.py and xd_option_seg.py
Customizable hyperparameters for model training and evaluation

## Installation
### Clone the repository
git clone https://github.com/your-repo/VadCLIP-multi.git
cd VadCLIP-multi

### Install required packages
pip install -r requirements.txt

## Datasets
The project supports two benchmark datasets:

UCF-Crime: Large-scale dataset for real-world anomaly detection in unconstrained environments
XD-Violence: Extreme domain violence detection dataset

## Usage
### Training on UCF-Crime Dataset
python ucf_train_seg.py --use-gcn False --use-scheme1 False

### Training on XD-Violence Dataset
python xd_train_seg.py --use-gcn False --use-scheme1 False

## Acknowledgments
We thank the authors of the original VadCLIP model and the creators of the UCF-Crime and XD-Violence datasets for making their resources publicly available.