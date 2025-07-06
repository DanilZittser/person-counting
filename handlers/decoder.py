import cv2
from typing import Generator
from handlers.handler import Handler
from handlers.models import Image


class VideoDecoder(Handler):
    """
    Компонент VAP: обработчик, который читает видео покадрово.
    """

    def __init__(self, video_path: str):
        """
        :param video_path: путь к видеофайлу
        """
        self.video_path: str = video_path
        self._cap: cv2.VideoCapture | None = None

    def on_start(self) -> None:
        """Открывает видеопоток"""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {self.video_path}")

    def handle(self) -> Generator[Image, None, None]:
        """Генератор кадров"""
        if self._cap is None:
            raise RuntimeError("VideoCapture не инициализирован. Вызовите on_start() перед handle().")

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def on_exit(self) -> None:
        """Освобождает ресурс VideoCapture"""
        if self._cap is not None:
            self._cap.release()
