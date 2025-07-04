from handlers.models import Box, Detection, Detections
from handlers.post_processor import PostProcessor

def manual_test_postprocessor():
    # Создаём пару детекций 
    det1 = Detection(box=Box(10, 10, 100, 100), score=0.95, label_as_int=0, label_as_str='person')
    det2 = Detection(box=Box(150, 150, 300, 300), score=0.85, label_as_int=1, label_as_str='car')

    detections_list = [det1, det2]

    # Создаём PostProcessor
    processor = PostProcessor()
    processor.on_start()
    result = processor.handle(detections_list)
    processor.on_exit()

    # Выводим результат
    print("Результаты обработки PostProcessor:")
    for det in result.detections:
        print(f"{det.label_as_str}: {det.score:.2f} — box({det.box.left}, {det.box.top}, {det.box.right}, {det.box.bottom})")

if __name__ == "__main__":
    manual_test_postprocessor()
