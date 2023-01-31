import os
import sys
import cv2
import base64
import tempfile
sys.path.append("../")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from imageai.Detection import VideoObjectDetection
from pathlib import Path
from PIL import Image
from utils import convert_video

st.title("Детекция на видео 📹")
st.text(os.listdir("../"))
st.text(os.listdir("./"))

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
            "/Users/nv27/Documents/Lovely ML/courses/DL School, part I/ObjectDetectionAPP/models/tiny-yolov3.pt"
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
    convert_video(
        input_path=os.path.join(execution_path, f"video_detected.mp4"),
        output_path=os.path.join(execution_path, f"video_detected_h264.mp4"),
    )
    st.markdown("### Видео с детектированными объектами")
    st.video(
        f"{execution_path}/video_detected_h264.mp4"
    )
