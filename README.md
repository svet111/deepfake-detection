# Deepfake Detection Project

This project implements a deepfake detection system for photos using a Convolutional Neural Network (CNN) based on ResNet50.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare the dataset:
   - Download datasets from the sources below.
   - Organize images into `data/train/real/`, `data/train/fake/`, `data/val/real/`, `data/val/fake/`, `data/test/real/`, `data/test/fake/`.

3. Train the model:
   ```
   python train.py
   ```

4. Evaluate the model:
   ```
   python evaluate.py
   ```

## Dataset Sources

- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **Deepfake Detection Challenge (DFDC)**: https://www.kaggle.com/c/deepfake-detection-challenge
- **NIST Media Forensics**: https://www.nist.gov/itl/iad/mig/media-forensics-challenge

For photo-specific datasets, you can extract frames from video datasets or use image forgery datasets like CASIA: http://forensics.idealtest.org/

Ensure to follow licensing and ethical guidelines when using these datasets.

## Status
This project provides a full pipeline for deepfake detection (training + evaluation), 
but due to hardware constraints, full-scale training on large datasets was not performed.
