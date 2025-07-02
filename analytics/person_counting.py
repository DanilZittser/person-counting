from typing import Any

from handlers.handler import Handler
from handlers.models import Blob, Detections, Events, Image, Tracks


class PersonCountingAnalytics:
    """Класс видеоаналитики по подсчёту проходящего пешеходного трафика на видео."""

    def __init__(
            self,
            pre_processor: Handler,  # подготовка кадра к передаче на вход нейронной сети
            inference: Handler,  # инференс нейронной сети
            post_processor: Handler,  # обработка результатов инференса (подготовка аннотированных объектов)
            tracker: Handler,  # построение маршрутов передвижения обнаруженных объектов
            heuristic: Handler,  # определение направления движения (слева направо или справа налево)
    ):
        self._pre_processor = pre_processor
        self._inference = inference
        self._post_processor = post_processor
        self._tracker = tracker
        self._heuristic = heuristic

    def on_start(self) -> None:
        self._pre_processor.on_start()
        ...

    def process_frame(self, image: Image) -> Events:
        """Основной метод видеоаналитики, реализующий логику обработки поступающих кадров."""
        blob: Blob = self._pre_processor.handle(image=image)
        inference_output: Any = self._inference.handle(blob=blob)
        detections: Detections = self._post_processor.handle(inference_output)
        finished_tracks: Tracks = self._tracker.handle(detections)
        events: Events = self._heuristic.handle(finished_tracks)
        return events

    def on_exit(self) -> None:
        self._pre_processor.on_exit()
        ...
