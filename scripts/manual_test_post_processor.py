import torch
from handlers.models import Detections
from handlers.post_processor import PostProcessor

class MockBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor([xyxy])  
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([cls])

class MockResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

def manual_test_postprocessor():
    boxes = [
        MockBox([10, 10, 100, 100], conf=0.95, cls=0),
        MockBox([150, 150, 300, 300], conf=0.85, cls=1)
    ]
    result = MockResult(boxes=boxes, names={0: 'person', 1: 'car'})

    processor = PostProcessor()
    processor.on_start()
    detections: Detections = processor.handle([result])
    processor.on_exit()

    print("Результаты обработки PostProcessor:")
    for det in detections.detections:
        print(f"{det.label_as_str}: {det.score:.2f} — box({det.box.left}, {det.box.top}, {det.box.right}, {det.box.bottom})")

if __name__ == "__main__":
    manual_test_postprocessor()

