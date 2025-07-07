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

    def handle(self, tensor: torch.Tensor) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call on_start() first.")

        # Конвертация тензора в numpy (HWC, RGB)
        img_np = tensor.squeeze(0).numpy()  
        img_np = np.transpose(img_np, (1, 2, 0))  
        img_np = (img_np * 255).astype(np.uint8)

        # Инференс
        results = self.model.predict(
            img_np,
            conf=self.conf_threshold,
            verbose=False
        )

        # Конвертация результатов в формат проекта
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        box=Box(left=x1, top=y1, right=x2, bottom=y2),
                        score=float(box.conf[0]),
                        label_as_int=int(box.cls[0]),
                        label_as_str=result.names[int(box.cls[0])]
                    )
                )
        return detections

    def on_exit(self):
        """Освобождение ресурсов модели."""
        if self.model is not None:
            del self.model
            self.model = None
