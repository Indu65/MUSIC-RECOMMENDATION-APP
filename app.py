import streamlit as st
import streamlit_webrtc as webrtc
import os
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Get absolute image path
image_path = os.path.join(os.getcwd(), "images", "logo.png")

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")
with col2:
    if os.path.exists(image_path):  
        st.image(image_path, width=530, use_column_width=True)
    else:
        st.error(f"⚠️ Image not found: {image_path}")
with col3:
    st.write("")

st.title("MyMusic")
st.write("MyMusic is an emotion detection-based music recommendation system. To get recommended songs, allow mic and camera access.")

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Mediapipe Setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Session state initialization
if "run" not in st.session_state:
    st.session_state["run"] = True

# Load detected emotion
if os.path.exists("detected_emotion.npy"):
    detected_emotion = np.load("detected_emotion.npy")[0]
else:
    detected_emotion = ""

if not detected_emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

# Emotion Detector Class
class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)  # Flip frame horizontally

        res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        # Store landmark data
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]
        print(pred)

        cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        np.save("detected_emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# Get user preferences
lang = st.text_input("Enter your preferred language")
artist = st.text_input("Enter your preferred artist")

if lang and artist and st.session_state["run"] is not False:
    webrtc_ctx = webrtc.webrtc_streamer(
        key="example",
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=EmotionDetector,  # Fix: Add video processing
    )

# Button to recommend music
btn = st.button("Recommend music")

if btn:
    if not detected_emotion:
        st.warning("Please let me capture your emotion first!")
        st.session_state["run"] = True  # Ensure WebRTC starts again
    else:
        search_query = f"{lang} {detected_emotion} songs {artist}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save("detected_emotion.npy", np.array([""]))  # Reset emotion
        st.session_state["run"] = False

st.write("Made by ❤ INDIRA")

# Hide Streamlit branding
st.markdown(
    """ <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style> """,
    unsafe_allow_html=True,
)
