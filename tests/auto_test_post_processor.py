import pytest
from handlers.models import Box, Detection, Detections
from handlers.post_processor import PostProcessor

def test_postprocessor_handle():
    det1 = Detection(box=Box(0, 0, 10, 10), score=0.9, label_as_int=1, label_as_str='person')
    det2 = Detection(box=Box(10, 10, 20, 20), score=0.8, label_as_int=2, label_as_str='car')
    detections_list = [det1, det2]

    processor = PostProcessor()
    processor.on_start()
    result = processor.handle(detections_list)
    processor.on_exit()

    assert isinstance(result, Detections)
    assert len(result.detections) == 2
    assert result.detections[0].label_as_str == 'person'
    assert result.detections[1].label_as_str == 'car'