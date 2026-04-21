# Pet Facial Emotion Recognition

A deep learning project that classifies pet facial expressions into emotion categories from images. The project compares multiple model architectures, including a baseline dense model, an advanced CNN, and an EfficientNet-based approach.

## Overview

This project aims to recognize pet emotions from facial images using computer vision and deep learning. It includes image preprocessing, model training, evaluation, and prediction on custom pet images.

The workflow covers:
- Loading and preprocessing pet facial images.
- Training deep learning models for emotion classification.
- Evaluating performance using accuracy, loss, confusion matrix, and classification report.
- Predicting emotions for new pet images using a saved model.

## Features

- Pet facial emotion classification from images.
- Multiple model experiments for comparison.
- Class weighting to handle imbalance.
- Learning-rate scheduling and early stopping.
- Accuracy and loss plots for training analysis.
- Confusion matrix and classification report for evaluation.
- Custom image prediction using a trained model.

## Models Used

- Baseline Dense Model.
- Advanced CNN.
- EfficientNet-based model.

## Dataset

The dataset contains pet facial expression images grouped into four emotion classes.  
All images are resized to `128 x 128` and normalized before training.

## Project Structure

- `pet_facial_emotion_AdvanceCNN.ipynb` — advanced CNN training and evaluation.
- `pet_facial_emotion_Basline_Dense.ipynb` — baseline dense model.
- `pet_facial_emotion_EfficientNet.ipynb` — EfficientNet experiment.
- `efficinetb5.ipynb` — EfficientNetB5 experiment.
- `cnnprediction.ipynb` — prediction and inference notebook.
- `predictenn.ipynb` — additional prediction notebook.
- `sigmoid_plotting.ipynb` — activation/plotting notebook.
- `README.md` — project documentation.

## Training Details

The advanced CNN model was trained with:
- Batch size: `32`
- Epochs: up to `100`
- Early stopping
- Learning-rate decay
- Class weights to improve learning on imbalanced classes

## Results

Example results from the advanced CNN experiment:
- Training accuracy: `73.75%`
- Validation accuracy: `42.50%`
- Test accuracy: `41.50%`
- Best validation accuracy: `43.13%`

## Inference

The saved model can be used to predict the emotion of a new pet image.  
The image is preprocessed, resized, normalized, and passed into the trained model for classification.

## Requirements

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn
- Pillow

## Installation

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn pillow
