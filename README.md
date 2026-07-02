# A Comparative Study of Vietnamese Scene Text Recognition Models

This project benchmarks and compares deep-learning approaches for **Vietnamese scene text recognition** — the task of detecting and reading text that appears in natural images (signboards, banners, product labels, etc.). Vietnamese is particularly challenging for OCR because of its dense use of stacked diacritics (tone marks combined with vowel modifiers), so the study pays special attention to how each model handles the Vietnamese alphabet.

## Demo

A video demo of the system is available here:

**[Demo on Google Drive](https://drive.google.com/drive/folders/1rqWXsUGQoBKmCCu_x6miaW6FkD8cB0Ee?hl=vi)**

## Overview

We study the full scene-text pipeline as two stages and compare different model families for each:

1. **Text detection** — locating text regions (polygons) in the image.
2. **Text recognition** — transcribing the cropped text region into a Vietnamese string.

All models are trained and evaluated on the **VinText** dataset, using a consistent Vietnamese alphabet defined with **NFD (decomposed) Unicode normalization** so that base letters and combining diacritics are handled explicitly.

## Dataset

https://www.kaggle.com/datasets/hungkhoi/vietnamese-scene-text-spotting-dataset-vintext

## Models Compared

| Approach | Detection | Recognition | Notes |
| --- | --- | --- | --- |
| **DBNet + CRNN** | DBNet (ResNet-18 backbone) | CRNN with CTC loss | Two-stage pipeline; detector is trained first, then frozen while CRNN is trained on word crops. |
| **PaddleOCRv3** | PP-OCRv3 mobile detector | PP-OCRv3 mobile recognizer | Similar to DBNet+CRNN in training method. |

## End-to-End Evaluation Protocol

To measure realistic performance, recognition is evaluated on crops produced by the **frozen detector** (not ground-truth boxes). Each predicted region is matched to the nearest ground-truth polygon at **IoU ≥ 0.5** to assign a label; unmatched predictions are discarded. Reported metrics include:

- **Detection:** Precision / Recall / H-mean
- **End-to-end:** Precision / Recall / H-mean
- **Recognition quality:** CER (Character Error Rate) and WER (Word Error Rate)

All experiments use a fixed seed (`SEED = 42`) for reproducibility.

## Repository Structure

```
.
├── Model-SUCCESSFUL/
│   ├── DBNet+CRNN/
│   │   ├── dbnet-detect-12epochs.ipynb              # train DBNet detector
│   │   ├── e2e-crnn-fixed.ipynb                     # CRNN recognizer
│   │   
│   └── PaddleOCRv3/
│       ├── PPOCRv3-detect-100epochs(curves).ipynb   # PaddleOCRv3 detector
│       └── ppocrv3-recog-50epochs.ipynb             # PaddleOCRv3 recognizer 
├── report.pdf                                       # Full written report
└── README.md
```

## Dataset

The experiments use the **VinText** Vietnamese scene-text dataset, which provides scene images with polygon-level text annotations and transcripts.

## Getting Started

The notebooks were developed to run on Kaggle/Colab-style GPU sessions. Core dependencies include:

```bash
pip install pyclipper shapely editdistance
# plus PyTorch (DBNet + CRNN) and PaddlePaddle / PaddleOCR (PaddleOCRv3)
```

Open the notebook for the model you want to reproduce and run the cells top-to-bottom. For the DBNet + CRNN pipeline, train the detector first (`dbnet-detect-12epochs.ipynb`) before training the recognizer.

## Report

See [`report.pdf`](report.pdf) for the full methodology, experimental setup, and detailed results discussion.
