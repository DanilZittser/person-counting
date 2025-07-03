import cv2
import numpy as np
import torch
import pytest
from handlers.pre_processor import PreProcessor
from handlers.inference import YoloInference

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
    """Тест работы YoloInference"""
    preprocessor = PreProcessor(input_size=(640, 640))
    detector = YoloInference(model_path="yolo11n.pt")
    
    tensor = preprocessor.handle(test_image)
    detector.on_start()
    detections = detector.handle(tensor)
    
    assert isinstance(detections, list)
    if detections:  
        det = detections[0]
        assert hasattr(det, 'box')
        assert hasattr(det, 'score')
    print(f"YoloInference: обнаружено {len(detections)} объектов")