import ultralytics
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Any, Optional
from handlers.handler import Handler
from handlers.models import Box, Detection 

class YoloInference(Handler):
    """Обработчик для инференса YOLO."""

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model: Optional[YOLO] = None

    def on_start(self):
        self.model = YOLO(self.model_path)

    def handle(self, tensor: torch.Tensor) -> Any:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call on_start() first.")

        # Прямой инференс по тензору
        results = self.model.predict(tensor, conf=self.conf_threshold, verbose=False)
        return results 

    def on_exit(self):
            self.model = None
