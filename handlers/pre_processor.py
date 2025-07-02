import os
import sys
import cv2
import numpy as np
import torch
from handlers.handler import Handler  

class PreProcessor(Handler):
    """
    Класс для подготовки изображения к подаче в модель YOLO v11.
    Преобразует BGR-изображение OpenCV в тензор формата (1, 3, 640, 640),
    с нормализованными значениями от 0 до 1.
    """

    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size

    def on_start(self):
        pass

    def handle(self, frame):
        # основной метод, который вызывается для обработки кадра
        resized = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img)
        return tensor

    def on_exit(self):
        pass
