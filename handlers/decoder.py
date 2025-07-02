import cv2
from handlers.handler import Handler


class VideoDecoder(Handler):
    """
    Компонент VAP: обработчик, который читает видео покадрово.
    """

    def __init__(self, video_path: str):
        """
        :param video_path: путь к видеофайлу
        """
        self.video_path = video_path
        self._cap = None

    def on_start(self):
        """Открывает видеопоток"""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {self.video_path}")

    def handle(self):
        """Генератор кадров"""
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def on_exit(self):
        """Освобождает ресурс VideoCapture"""
        if self._cap is not None:
            self._cap.release()
