import pytest
import torch
from handlers.post_processor import PostProcessor
from handlers.models import Detections

class MockBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor([xyxy])  
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([cls])

class MockResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

def test_postprocessor_handle():
    boxes = [
        MockBox([0, 0, 10, 10], conf=0.9, cls=0),
        MockBox([20, 20, 40, 40], conf=0.85, cls=2)
    ]
    result = MockResult(boxes=boxes, names={0: 'person', 2: 'car'})

    processor = PostProcessor()
    processor.on_start()
    detections = processor.handle([result])
    processor.on_exit()

    assert isinstance(detections, Detections)
    assert len(detections.detections) == 2
    assert detections.detections[0].label_as_str == 'person'
    assert detections.detections[1].label_as_str == 'car'