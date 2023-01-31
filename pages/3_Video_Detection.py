import os
import sys
import cv2
import base64
import tempfile
sys.path.append("../")
sys.path.append("./")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from imageai.Detection import VideoObjectDetection
from pathlib import Path
from PIL import Image

def convert_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.write("frame_params")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    st.write("out")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    st.write("success")
    cap.release()
    out.release()
    st.write("success 2")


st.title("Детекция на видео 📹")
st.markdown("----------------------------")
st.markdown("### Инструкция по применению")
st.text("1. Выбрать размер модели для применения (см. подсказку)")
st.text("2. Загрузить видео для детекции")
st.text("3. Выбрать параметры видео")
st.text("4. Запустить детекцию")
st.markdown("----------------------------")

help_msg = (
    "Размер модели влияет на скорость и точность распознавания: \n"
    "чем сложнее модель, тем точнее она распознает объекты, но медленее работает."
)
model_size = st.sidebar.selectbox(
    "Размер модели для детекции",
    ("TinyYOLOv3", "YOLOv3", "RetinaNet"),
    help=help_msg
)
help_msg = (
    "Видео в формате .mp4. "
    "Размер видео ограничен 200Mb."
)
uploaded_file = st.sidebar.file_uploader(
    "Выбор видео для обработки",
    type=["mp4"],
    help=help_msg
)
help_msg = (
    "Кадры в секунду или fps (frames per second) — "
    "это количество кадров, которые камера фиксирует за одну секунду."
)
frames_per_second = st.sidebar.slider(
    label="Кадры в секунду", min_value=1, max_value=240, value=30, step=1, help=help_msg
)
help_msg = (
    "Частота детекции кадров - "
    "это интервалы кадров, который будут детектированы."
)
frame_detection_interval = st.sidebar.slider(
    label="Частота детекции кадров", min_value=1, max_value=240, value=1, step=1, help=help_msg
)

if uploaded_file:
    with st.spinner(text="Загрузка видео..."):
        st.markdown("### Загруженное видео для детектирования")
        st.video(uploaded_file)

if uploaded_file:
    start_detection = st.sidebar.checkbox("Запустить детекцию объектов.")
    try:
        os.mkdir("./video/")
    except FileExistsError:
        pass

    with open("./video/video_for_detect.mp4", mode='wb') as w:
        w.write(uploaded_file.getvalue())

    detector = VideoObjectDetection()
    if model_size == "TinyYOLOv3":
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(
            "./models/tiny-yolov3.pt"
        )
    elif model_size == "YOLOv3":
        st.write("В бесплатной версии приложения этот тип модели не поддерживается 🙂")
        #detector.setModelTypeAsYOLOv3()
        #detector.setModelPath(
        #    "./models/yolov3.pt"
        #)
    elif model_size == "RetinaNet":
        st.write("В бесплатной версии приложения этот тип модели не поддерживается 🙂")
        #detector.setModelTypeAsRetinaNet()
        #detector.setModelPath(
        #    "./models/retinanet_resnet50_fpn_coco-eeacb38b.pth"
        #)
    try:
        detector.loadModel()
    except ValueError:
        st.write("Для детекции объектов нужно выбрать модель TinyYOLOv3.")

    execution_path = "./video/"
    detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, "video_for_detect.mp4"),
        output_file_path=os.path.join(execution_path, f"video_detected"),
        frame_detection_interval=frame_detection_interval,
        frames_per_second=frames_per_second,
        display_percentage_probability=True,
        log_progress=True,
    )
    st.write(os.listdir("./video/"))
    convert_video(
        input_path=os.path.join(execution_path, f"video_detected.mp4"),
        output_path="./video/video_detected_h264.mp4",
    )
    st.write(os.listdir("./video/"))
    st.write(os.listdir(execution_path))
    st.write(os.listdir("./"))
    st.write(os.listdir("./pages/"))
    st.markdown("### Видео с детектированными объектами")

    try:
        st.video(
            "./video/video_detected_h264.mp4"
        )
    except FileNotFoundError:
        st.video(
            f"{execution_path}video_detected.mp4"
        )

