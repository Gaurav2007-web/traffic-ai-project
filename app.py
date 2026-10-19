import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic AI", layout="wide")
st.title("🚦 Smart Emergency Vehicle & Dynamic Traffic Optimizer")
st.markdown("Automated Green Corridor & Adaptive Signal Duration System")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()
VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Traffic CCTV Video", type=["mp4", "avi", "mov"])
ambulance_override = st.sidebar.checkbox("🚨 Emergency Vehicle Detected (Ambulance)", value=False)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Detection Stream")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Signal Analytics")
    status_placeholder = st.empty()
    metric_vehicles = st.empty()
    timer_placeholder = st.empty()

if uploaded_file is not None:
    temp_video_path = "temp_input_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture(temp_video_path)
    base_time = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False)
        vehicle_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, VEHICLE_CLASSES[cls_id], (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if ambulance_override:
            signal_color = "🟢 EMERGENCY GREEN (CORRIDOR OPEN)"
            allocated_time = "MAX / PRIORITY CLEAR"
            cv2.putText(frame, "EMERGENCY OVERRIDE: GREEN CORRIDOR", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            allocated_time = f"{base_time + (vehicle_count * 2)} Seconds"
            if vehicle_count > 5:
                signal_color = "🟢 HIGH DENSITY GREEN"
            elif vehicle_count > 0:
                signal_color = "🟡 NORMAL GREEN"
            else:
                signal_color = "🔴 RED (EMPTY LANE)"

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        status_placeholder.markdown(f"### Current State:\n**{signal_color}**")
        metric_vehicles.metric(label="Vehicles Detected", value=vehicle_count)
        timer_placeholder.metric(label="Allocated Green Time", value=allocated_time)

    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
else:
    st.info("👈 Left sidebar se traffic video upload karein.")
