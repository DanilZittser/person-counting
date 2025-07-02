# библиотеки 
import cv2
import numpy as np
import torch

class PreProcessor:
    """
    Класс для подготовки изображения к подаче в модель YOLO v11.
    Преобразует BGR-изображение OpenCV в тензор формата (1, 3, 640, 640),
    с нормализованными значениями от 0 до 1.
    """

    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size  # Размер входа модели YOLO (width, height)

    def __call__(self, frame):
        """
        Вызывается при передаче кадра. Возвращает тензор, готовый для YOLO.
        """
        return self.preprocess(frame)

    def preprocess(self, frame):
        # 1. Изменение размера изображения
        resized = cv2.resize(frame, self.input_size)

        # 2. Перевод BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Нормализация значений пикселей в диапазон [0, 1]
        img = rgb.astype(np.float32) / 255.0

        # 4. HWC -> CHW (для PyTorch)
        img = np.transpose(img, (2, 0, 1))

        # 5. Добавление batch dimension (1, 3, H, W)
        img = np.expand_dims(img, axis=0)

        # 6. Преобразование в PyTorch тензор
        tensor = torch.from_numpy(img)

        return tensor
