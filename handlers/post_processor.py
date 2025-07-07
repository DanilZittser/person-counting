from handlers.handler import Handler
from handlers.models import Box, Detection, Detections

class PostProcessor(Handler):
    """
    Преобразует результаты инференса YOLO (список результатов) в объект Detections.
    Каждый результат содержит список боксов с координатами, уверенностью и классами.
    """

    def on_start(self):
        pass

    def handle(self, results: list) -> Detections:
        detections = []

        for result in results:
            if not hasattr(result, 'boxes') or not hasattr(result, 'names'):
                continue  

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        box=Box(left=x1, top=y1, right=x2, bottom=y2),
                        score=float(box.conf[0]),
                        label_as_int=int(box.cls[0]),
                        label_as_str=result.names[int(box.cls[0])]
                    )
                )

        return Detections(detections=detections)

    def on_exit(self):
        pass