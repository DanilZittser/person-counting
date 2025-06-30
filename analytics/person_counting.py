from handlers.handler import Handler
from handlers.models import Blob, Image, Event


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
        ...

    def on_start(self) -> None:
        self._pre_processor.on_start()
        ...

    def process_frame(self, image: Image) -> Event:
        """Основной метод видеоаналитики, реализующий логику обработки поступающих кадров."""
        blob: Blob = self._pre_processor.handle(image=image)
        ...
        return event

    def on_exit(self) -> None:
        self._pre_processor.on_exit()
        ...