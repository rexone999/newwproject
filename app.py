import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Download model if not present
if not os.path.exists("vlstm_92.h5"):
    import gdown
    url = "https://drive.google.com/uc?id=1pXOBqK6zSgnXHweIEt-xt747Vgr5Pfmw"
    gdown.download(url, "vlstm_92.h5", fuzzy=True, use_cookies=True)

from keras.models import load_model
from keras.layers import TimeDistributed, LSTM, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

st.title("🎬 Violence Detection in Video")
st.write("Upload a video and the app will detect violent content and show timestamps.")

# Load model with necessary custom layers
custom_objects = {
    'TimeDistributed': TimeDistributed,
    'LSTM': LSTM,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Flatten': Flatten,
    'Dense': Dense,
    'Dropout': Dropout
}
model = load_model("vlstm_92.h5", custom_objects=custom_objects)

# Extract frames at intervals and timestamps
def extract_frames(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            resized_frame = cv2.resize(frame, (224, 224))
            frames.append(resized_frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        frame_count += 1

    cap.release()
    return np.array(frames), timestamps

# Predict violence per frame
def predict_violence(frames, timestamps, threshold=0.5):
    X = frames / 255.0
    X = X.reshape(X.shape[0], 224, 224, 3)
    preds = model.predict(X, verbose=0)
    violence_times = [round(timestamps[i], 2) for i, pred in enumerate(preds) if pred > threshold]
    return violence_times

# Upload interface
uploaded_file = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    st.write("⏳ Processing video...")
    frames, timestamps = extract_frames(tfile.name)
    violence_times = predict_violence(frames, timestamps)

    if violence_times:
        st.success("🚨 Violence detected at:")
        st.write(violence_times)
    else:
        st.success("✅ No violence detected in the video.")
