import cv2
from handlers.pre_processor import PreProcessor
from handlers.inference import YoloInference

# Инициализация
preprocessor = PreProcessor(input_size=(640, 640))
detector = YoloInference(model_path="yolo11n.pt")

# Загрузка изображения
image = cv2.imread("/content/person-counting/result.jpg")  
tensor = preprocessor.handle(image)  
detector.on_start()  
detections = detector.handle(tensor)  

# 4. Визуализация результатов
for det in detections:
    x1, y1, x2, y2 = det.box.left, det.box.top, det.box.right, det.box.bottom
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{det.label_as_str} {det.score:.2f}", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

from google.colab.patches import cv2_imshow
cv2_imshow(image) автомат тест import cv2
