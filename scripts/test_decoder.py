# "Ручной" тест

import sys
import os


# Добавляем корневую директорию проекта в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from handlers.decoder import VideoDecoder

video_path = r"C:\Users\LENOVO\Desktop\Видосик.mp4"

decoder = VideoDecoder(video_path)
decoder.on_start()

for frame in decoder.handle():
    print(frame.shape)

decoder.on_exit()
