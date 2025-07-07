import cv2
import numpy as np
import torch
import pytest

from handlers.pre_processor import PreProcessor
from handlers.inference import YoloInference
from handlers.post_processor import PostProcessor  
from handlers.models import Detections  

@pytest.fixture
def test_image():
    """Фикстура с тестовым изображением"""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    return img

def test_preprocessor_output(test_image):
    """Тест преобразования изображения в тензор"""
    preprocessor = PreProcessor(input_size=(640, 640))
    tensor = preprocessor.handle(test_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 640, 640)
    assert 0 <= tensor.min() <= 1
    assert 0 <= tensor.max() <= 1
    print("PreProcessor: преобразование изображения в тензор работает корректно")

def test_detector_output(test_image):
    """Интеграционный тест инференса и постпроцессинга"""
    preprocessor = PreProcessor(input_size=(640, 640))
    detector = YoloInference(model_path="yolo11n.pt")
    postprocessor = PostProcessor()

    tensor = preprocessor.handle(test_image)
    detector.on_start()
    raw_predictions = detector.handle(tensor)  

    detections = postprocessor.handle(raw_predictions)  

    assert isinstance(detections, Detections)
    assert hasattr(detections, 'detections')
    assert isinstance(detections.detections, list)

    for det in detections.detections:
        assert hasattr(det, 'box')
        assert hasattr(det, 'score')
        assert 0.0 <= det.score <= 1.0

    print(f"YoloInference + PostProcessor: обнаружено {len(detections.detections)} объектов")