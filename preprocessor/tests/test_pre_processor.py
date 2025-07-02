from handlers.pre_processor import PreProcessor
frame = cv2.imread("/content/тест.jpeg")

# Проверка
if frame is None:
    print("Ошибка: не удалось загрузить изображение.")
else:
    print(f"Исходный размер: {frame.shape}")
    pre = PreProcessor()
    tensor = pre(frame)

    print("Форма тензора:", tensor.shape)
    print("Мин значение:", tensor.min().item())
    print("Макс значение:", tensor.max().item())


    