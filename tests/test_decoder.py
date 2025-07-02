# Автоматические тесты

import os
import pytest
import cv2

from handlers.decoder import VideoDecoder

TEST_VIDEO_PATH = r"C:\Users\LENOVO\Desktop\Видосик.mp4"


@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH), reason="Видео не найдено")
def test_video_decoder_handle_returns_frames():
    decoder = VideoDecoder(TEST_VIDEO_PATH)
    decoder.on_start()

    generator = decoder.handle()
    frame = next(generator, None)

    assert frame is not None, "handle() должен вернуть хотя бы один кадр"
    assert isinstance(frame, (list, tuple, cv2.Mat, type(frame))), "Кадр должен быть изображением"
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
