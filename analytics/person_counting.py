from handlers.handler import Handler
from handlers.models import Events, Image


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

    def on_start(self):
        for component in [self._pre_processor, self._inference,
                          self._post_processor, self._tracker, self._heuristic]:
            component.on_start()

    def process_frame(self, image: Image) -> Events:
        """Основной метод видеоаналитики, реализующий логику обработки поступающих кадров."""
        tensor = self._pre_processor.handle(image)
        raw_results = self._inference.handle(tensor)
        detections = self._post_processor.handle(raw_results)
        finished_tracks = self._tracker.handle(detections.detections)
        events = self._heuristic.handle(finished_tracks)
        return Events(events)

    def on_exit(self):
        for component in [self._pre_processor, self._inference,
                          self._post_processor, self._tracker, self._heuristic]:
            component.on_exit()
