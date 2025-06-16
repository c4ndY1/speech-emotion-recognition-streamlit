import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os

@st.cache_resource
def load_model_components():
    scaler = joblib.load("model/scaler.pkl")
    selector = joblib.load("model/feature_selector.pkl")
    le = joblib.load("model/label_encoder.pkl")
    model = load_model("model/emotion_model.h5")
    return scaler, selector, le, model


# Set Streamlit page config
st.set_page_config(
    page_title="Speech Emotion Classifier",
    page_icon="🎧",
    layout="centered",
)

# Load preprocessing tools and model
#scaler = joblib.load("model/scaler.pkl")
#selector = joblib.load("model/feature_selector.pkl")
#le = joblib.load("model/label_encoder.pkl")
#model = load_model("model/emotion_model.h5")
scaler, selector, le, model = load_model_components()

# Helper: Feature summarization
def summarize_feature(feature):
    return np.hstack([
        np.mean(feature.T, axis=0),
        np.std(feature.T, axis=0),
        np.min(feature.T, axis=0),
        np.max(feature.T, axis=0)
    ])

# Helper: Feature extraction
def features_extract(path):
    try:
        audio, sr = librosa.load(path, sr=22050, res_type='scipy')
        stft = np.abs(librosa.stft(audio))

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        if mfccs.shape[1] == 0: return None

        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)

        if any(feat.shape[1] == 0 for feat in [chroma, contrast, mel, tonnetz]):
            return None

        combined = np.hstack([
            summarize_feature(np.vstack([mfccs, mfccs_delta, mfccs_delta2])),
            summarize_feature(chroma),
            summarize_feature(contrast),
            summarize_feature(mel),
            summarize_feature(tonnetz),
            summarize_feature(zcr),
            summarize_feature(rms),
        ])
        return combined
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# UI: Title and instructions
st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>Speech Emotion Recognition</h1>"
    "<p style='text-align: center;'>Upload a .wav file to detect the emotion</p><hr>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing emotion..."):
        features = features_extract("temp_audio.wav")
        if features is not None:
            features_scaled = scaler.transform([features])
            features_selected = selector.transform(features_scaled)
            prediction = model.predict(features_selected)
            predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
            confidence = prediction[0][np.argmax(prediction)] * 100

            st.markdown("---")
            st.markdown(
                f"<h2 style='text-align: center; font-size: 40px;'>{predicted_class.upper()}</h2>"
                f"<p style='text-align: center; font-size: 18px;'>Confidence: {confidence:.2f}%</p>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Unable to extract meaningful features. Try a clearer audio.")

    os.remove("temp_audio.wav")
else:
    st.info("Upload a valid `.wav` audio file.")

