import cv2
from handlers.pre_processor import PreProcessor  

def main():
    image_path = "/content/тест.jpeg"
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Ошибка: не удалось загрузить изображение по пути {image_path}")
        return

    print(f"Исходный размер изображения: {frame.shape}")

    preprocessor = PreProcessor()
    tensor = preprocessor.handle(frame)  

    print(f"Форма выходного тензора: {tensor.shape}")
    print(f"Минимальное значение в тензоре: {tensor.min().item()}")
    print(f"Максимальное значение в тензоре: {tensor.max().item()}")

if __name__ == "__main__":
    main()
