import cv2
from handlers.pre_processor import PreProcessor
from handlers.inference import YoloInference
from handlers.post_processor import PostProcessor
from google.colab.patches import cv2_imshow  

def main():
    # Инициализация компонентов
    preprocessor = PreProcessor(input_size=(640, 640))
    detector = YoloInference(model_path="yolo11n.pt")
    postprocessor = PostProcessor()

    # Загрузка изображения
    image = cv2.imread("/content/IMG_20211217_171157.jpg")
    image = cv2.resize(image, (640, 640))
    if image is None:
        print("Ошибка: изображение не найдено.")
        return

    # Обработка
    tensor = preprocessor.handle(image)
    detector.on_start()
    raw_detections = detector.handle(tensor)
    detections = postprocessor.handle(raw_detections) 

    # Визуализация
    for det in detections.detections:
        x1, y1, x2, y2 = det.box.left, det.box.top, det.box.right, det.box.bottom
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image,f"{det.label_as_str} {det.score:.2f}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2,)
    cv2_imshow(image)

if __name__ == "__main__":
    main()