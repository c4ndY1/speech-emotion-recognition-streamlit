
# Speech Emotion Recognition (SER) Web App

This project implements a **Speech Emotion Recognition** system using **deep learning** on extracted audio features. The application allows users to upload `.wav` audio files, and the system predicts the **dominant emotion** in the speech. A **Streamlit frontend** is built for clean and interactive deployment.

---

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Streamlit Deployment](#streamlit-deployment)
- [File Structure](#file-structure)
- [Usage Instructions](#usage-instructions)
- [Future Improvements](#future-improvements)

---

## Project Description

This is a **machine learning-based web application** that predicts human emotions from audio speech. It classifies audio into **eight emotions**:
- `angry`
- `calm`
- `disgust`
- `fearful`
- `happy`
- `neutral`
- `sad`
- `surprised`

The model is trained on extracted features from audio clips and is optimized for high validation accuracy and generalization.

---

## Dataset

The app uses audio files from the **RAVDESS dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song). This dataset contains `.wav` files of actors speaking in various emotional tones. Only the speech portions were used (not the singing ones).

---

## Preprocessing Pipeline

1. **Audio Loading:** `.wav` files are loaded using `librosa` with a sampling rate of 22050 Hz.
2. **Feature Extraction:**
   - **MFCCs**
   - **Chroma features**
   - **Spectral Contrast**
   - **Tonnetz**
3. **DataFrame Creation:** Extracted features are converted to a single row vector for each audio sample and stored in a `pandas` DataFrame with labels.

4. **Imbalance Handling:**
   - Original dataset had uneven label distributions.
   - Applied **upsampling** using `sklearn.utils.resample` to balance all 8 emotion classes.

---

## Feature Selection

1. **Scaling:** StandardScaler is applied to normalize all features.
2. **Label Encoding:** Emotions are encoded using `LabelEncoder`.
3. **One-Hot Encoding:** Categorical labels are converted for model training.
4. **Feature Selection:** Top 300 features selected using `SelectKBest` with `f_classif`.

---

## Model Architecture

```
Input: 300 selected features

Dense(512, relu, L2 regularization)
→ BatchNorm + Dropout(0.4)
→ Dense(256, relu, L2)
→ BatchNorm + Dropout(0.3)
→ Dense(112, relu, L2)
→ Dropout(0.2)
→ Output: Dense(num_classes, softmax)
```

- Optimizer: Adam (`lr=5e-4`)
- Loss Function: `categorical_crossentropy`
- Callbacks:
  - `EarlyStopping` (patience=20)
  - `ReduceLROnPlateau` (factor=0.2, patience=10)
- Class Weights: Computed using `compute_class_weight()` for imbalance.

---

## Evaluation Metrics

- Accuracy on Test Set: **[Insert actual % here, e.g., 87.25%]**
- Classification Report:
  - Precision, Recall, and F1-score per emotion.
- Confusion Matrix: Visual analysis of model predictions.

---

## Streamlit Deployment

The app is deployed via a **Streamlit interface** with the following frontend decisions:

- Clean, minimal UI without sidebar.
- Prominent emotion output using large fonts and icons (no emojis).
- Upload or record `.wav` file directly.
- Displays probability scores in descending order for top-3 emotions.
- No charts or unnecessary UI clutter.

---

## File Structure

```
project/
│
├── app.py                     # Streamlit app
├── features_extract.py        # Custom function for extracting audio features
├── processed_audio_features.pkl  # Balanced preprocessed data
│
├── model/
│   ├── emotion_model.h5       # Trained Keras model
│   ├── scaler.pkl             # StandardScaler
│   ├── selector.pkl           # SelectKBest for 300 features
│   └── label_encoder.pkl      # LabelEncoder for emotions
│
└── requirements.txt           # List of required Python packages
```

---

## Usage Instructions

### 1. Clone and Set Up

```bash
git clone https://github.com/yourusername/speech-emotion-app.git
cd speech-emotion-app
pip install -r requirements.txt
```

### 2. Launch the App

```bash
streamlit run app.py
```

### 3. Interact

- Upload a `.wav` audio file.
- View the predicted emotion and probabilities.

---

## Future Improvements

- Add support for longer or real-time audio.
- Replace `SelectKBest` with PCA or deep autoencoder-based feature reduction.
- Use CRNN or transformer-based models for better temporal modeling.
- Add user authentication for personalized emotion tracking.

---

## Requirements

- Python ≥ 3.8
- `librosa`
- `scikit-learn`
- `tensorflow`
- `numpy`, `pandas`, `matplotlib`, `streamlit`, `joblib`
