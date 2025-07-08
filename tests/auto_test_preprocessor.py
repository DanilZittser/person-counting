import numpy as np
import torch
from handlers.pre_processor import PreProcessor  

def test_preprocessor_output_shape_and_range():
    dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    
    preprocessor = PreProcessor()
    tensor = preprocessor.handle(dummy_frame)
    
    assert tensor.shape == (1, 3, 640, 640)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
