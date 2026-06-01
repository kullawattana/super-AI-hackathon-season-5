# Super AI Hackathon Season 5 Notebooks

This repository contains exploratory notebooks, baseline models, and evaluation utilities for multiple Super AI Hackathon Season 5 challenges. The work is organized around Kaggle-style competitions and includes experiments for tabular data, signal processing, IoT sensor classification, medical prediction, image processing, and Thai image captioning.

## Project Overview

The notebooks cover several challenge domains:

- **Signal Processing / FRB Detection**: spectrogram/image-based modeling with PyTorch, `timm`, OpenCV, and FFT preprocessing.
- **IoT Sleep Stage Classification**: time-series classification from wearable sensor features using TensorFlow/Keras LSTM and GRU models.
- **Heart Disease Recognition**: tabular classification with feature engineering and preprocessing experiments.
- **Data Science Income Prediction**: tabular income prediction using preprocessing, label encoding, scaling, and neural network models.
- **Image Captioning / NLP**: Thai image captioning experiments and BLEU-based evaluation utilities.
- **Image Processing**: dataset loading, preprocessing, and baseline model experiments for image-based tasks.

Most notebooks were designed to run in a Kaggle notebook environment, so dataset paths usually point to `/kaggle/input/...`.

## Repository Structure

```text
.
├── hackathon-super-ai-ss-5-domain-ds.ipynb
├── hackathon-super-ai-ss-5-image-captioning.ipynb
├── hackathon-super-ai-ss-5-signal-processing-v1.ipynb
├── ss5_frb_detection_500320_1.ipynb
├── ss5_frb_detection_500320_2.ipynb
├── ss5_heart_prediction_500320.ipynb
├── ss5_sleep_staging_500320.ipynb
└── Hackathon/
    ├── BaseLineModel.ipynb
    ├── EvaluationMetric.ipynb
    ├── ScoreFunction.ipynb
    ├── SS5_DS_Baseline.ipynb
    ├── SS5_NLP_Baseline.ipynb
    ├── hack2-image-processing.ipynb
    ├── hackathon-super-ai-ss-5-iot-sleepstage.ipynb
    └── image captioning.txt
```

## Notebook Guide

| Notebook | Purpose |
| --- | --- |
| `hackathon-super-ai-ss-5-signal-processing-v1.ipynb` | Signal processing baseline for FRB-style detection using image/spectrogram preprocessing and deep learning. |
| `ss5_frb_detection_500320_1.ipynb` | FRB detection experiment variant. |
| `ss5_frb_detection_500320_2.ipynb` | Additional FRB detection experiment variant. |
| `ss5_sleep_staging_500320.ipynb` | Sleep stage classification using sensor time-series data and Keras LSTM models. |
| `ss5_heart_prediction_500320.ipynb` | Heart disease prediction workflow with tabular preprocessing and feature engineering. |
| `hackathon-super-ai-ss-5-domain-ds.ipynb` | Income prediction / domain data science notebook using tabular preprocessing and neural networks. |
| `hackathon-super-ai-ss-5-image-captioning.ipynb` | Image captioning environment setup and model experimentation with Hugging Face tools. |
| `Hackathon/SS5_DS_Baseline.ipynb` | Data science baseline using GRU-based modeling and preprocessing. |
| `Hackathon/SS5_NLP_Baseline.ipynb` | NLP baseline with Thai language tooling such as PyThaiNLP and spaCy. |
| `Hackathon/BaseLineModel.ipynb` | Image-based PyTorch baseline model. |
| `Hackathon/EvaluationMetric.ipynb` | BLEU scoring utility for captioning submissions. |
| `Hackathon/ScoreFunction.ipynb` | Custom scoring function for object/polygon-style predictions. |
| `Hackathon/hack2-image-processing.ipynb` | Image loading, preprocessing, and model training workflow. |
| `Hackathon/hackathon-super-ai-ss-5-iot-sleepstage.ipynb` | Multiple sleep-stage classification experiments with reported trial scores. |

## Environment

The notebooks use a mix of Python data science, machine learning, NLP, and computer vision libraries. Common dependencies include:

- Python 3.10+
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- OpenCV
- Pillow
- SciPy
- TensorFlow / Keras
- PyTorch
- torchvision
- timm
- tqdm
- transformers
- accelerate
- bitsandbytes
- spaCy
- PyThaiNLP
- NLTK

Some notebooks install their own dependencies inside the notebook using `pip`. If running locally, install the needed packages before executing each notebook.

Example local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python pillow scipy tensorflow torch torchvision timm tqdm transformers accelerate nltk pythainlp spacy
```

GPU acceleration is recommended for the deep learning notebooks, especially the FRB detection, image processing, and image captioning experiments.

## Datasets

Datasets are not included in this repository. Most notebooks expect Kaggle input directories such as:

- `/kaggle/input/data-science-income-prediction/`
- `/kaggle/input/hearth-disease-recognition/`
- `/kaggle/input/io-t-sleep-stage-classification-version-2/`
- `/kaggle/input/image-processing-thai-language-image-captioning/`

When running outside Kaggle, download the required competition datasets and update the path variables inside the relevant notebooks.

## How to Run

1. Open the target notebook in Kaggle, Jupyter Notebook, JupyterLab, or VS Code.
2. Make sure the required dataset is available at the path used in the notebook.
3. Install any notebook-specific dependencies.
4. Run cells from top to bottom.
5. Export the generated predictions or submission file according to the competition format.

## Notes

- Several notebooks are experiment variants and may contain hardcoded paths, trial-specific comments, or Kaggle-specific assumptions.
- Evaluation notebooks in `Hackathon/` are intended to validate submission formats and compute challenge-specific scores.
- The repository is primarily a research and competition workspace, not a packaged Python application.
