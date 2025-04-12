
import streamlit as st
import cv2
import numpy as np
import tempfile
from keras.models import load_model
import os
import os
import os
if not os.path.exists("vlstm_92.h5"):
    import gdown
    file_id = "1pXOBqK6zSgnXHweIEt-xt747Vgr5Pfmw"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "vlstm_92.h5", quiet=False)


st.title("Violence Detection in Video")
st.write("Upload a video and the app will detect violence and generate timestamps.")

model = load_model("vlstm_92.h5")

def extract_frames(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
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

def predict_violence(frames, timestamps, threshold=0.5):
    X = frames / 255.0
    X = X.reshape(X.shape[0], 224, 224, 3)
    preds = model.predict(X, verbose=0)
    violence_times = [round(timestamps[i], 2) for i, pred in enumerate(preds) if pred > threshold]
    return violence_times

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)
    
    st.write("Processing video...")
    frames, timestamps = extract_frames(tfile.name)
    violence_times = predict_violence(frames, timestamps)
    
    if violence_times:
        st.success("Violence detected at the following timestamps (in seconds):")
        st.write(violence_times)
    else:
        st.success("No violence detected in the video.")
