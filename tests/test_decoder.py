# Автоматические тесты

import os
import pytest
import numpy as np
from handlers.decoder import VideoDecoder

# Абсолютный путь к видеофайлу
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_VIDEO_PATH = os.path.join(project_root, "assets", "tests", "video_decoder", "Видосик.mp4")


@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH), reason="Видео не найдено")
def test_video_decoder_handle_returns_frames():
    decoder = VideoDecoder(TEST_VIDEO_PATH)
    decoder.on_start()

    generator = decoder.handle()
    frame = next(generator, None)

    assert frame is not None, "handle() должен вернуть хотя бы один кадр"
    assert isinstance(frame, np.ndarray), "Кадр должен быть изображением (np.ndarray)"
    assert hasattr(generator, '__iter__'), "handle() должен быть генератором"

    decoder.on_exit()


@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH), reason="Видео не найдено")
def test_video_decoder_lifecycle():
    decoder = VideoDecoder(TEST_VIDEO_PATH)
    assert decoder._cap is None, "До on_start _cap должен быть None"

    decoder.on_start()
    assert decoder._cap is not None and decoder._cap.isOpened(), "После on_start _cap должен быть открыт"

    decoder.on_exit()
    assert decoder._cap is not None and not decoder._cap.isOpened(), "После on_exit _cap должен быть закрыт"
