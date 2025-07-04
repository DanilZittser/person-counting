from handlers.handler import Handler
from handlers.models import Detection, Detections

class PostProcessor(Handler):
    """
    Обработчик, преобразующий список Detection в объект Detections.
    """
    def on_start(self):
        pass

    def handle(self, detections: list[Detection]) -> Detections:
        return Detections(detections=detections)

    def on_exit(self):
        pass