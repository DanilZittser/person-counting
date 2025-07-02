# PreProcessor для YOLO v11

Этот обработчик предназначен для подготовки декодированных кадров к подаче в модель YOLO v11.

## Основные возможности
- Изменение размера изображения до 640x640 пикселей (требование модели)
- Преобразование цветового пространства BGR → RGB
- Нормализация значений пикселей в диапазоне [0, 1]
- Преобразование numpy-массива в тензор PyTorch формата `(1, 3, 640, 640)`
- Готовность к пакетной обработке и подаче в модель YOLO v11

## Использование

```python
from handlers.pre_processor import PreProcessor
import cv2

pre = PreProcessor()
frame = cv2.imread("path/to/image.jpg")
tensor = pre(frame)

print(tensor.shape)  # torch.Size([1, 3, 640, 640])
