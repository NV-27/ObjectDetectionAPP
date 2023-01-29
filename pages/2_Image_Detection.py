import cv2
import sys
sys.path.append("../")

import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
#from utils import get_model

st.title("Детекция на изображениях 🖼️")

st.markdown("----------------------------")
st.markdown("### Инструкция по применению")
st.text("1. Выбрать размер модели для применения (см. подсказку)")
st.text("2. Загрузить одно или несколько изображений.")
st.markdown("----------------------------")

help_msg = (
    "Размер модели влияет на скорость и точность распознавания: \n"
    "чем сложнее модель, тем точнее она распознает объекты, но медленее работает."
)
model_size = st.sidebar.selectbox(
    "Размер модели для детекции",
    ("n", "s", "m", "l", "x"),
    help=help_msg
)
help_msg = (
    "Изображение в формате .png, .jpg, .jpeg. "
    "Допускается загрузка нескольких изображений сразу."
)
uploaded_file = st.sidebar.file_uploader(
    "Выбор картинки для обработки",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help=help_msg
)
@st.cache
def get_model(model_name: str):
    model = torch.hub.load(
        "ultralytics/yolov5", model=f"yolov5{model_name}", pretrained=True
    )
    return model


if uploaded_file:
    show_image_checkbox = st.sidebar.checkbox("Показать входные изображения.")
    
    file_bytes = np.asarray(
        bytearray(uploaded_file[0].read()), dtype=np.uint8
    )
    opencv_image = cv2.imdecode(file_bytes, 1)

    if show_image_checkbox:
        st.image(opencv_image)

    model = get_model(model_size)
    results = model(opencv_image)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(results.render()[0])
    st.pyplot(fig)
