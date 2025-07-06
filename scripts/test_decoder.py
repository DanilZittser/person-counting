import sys
import os

# Добавляем корневую директорию проекта в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from handlers.decoder import VideoDecoder


def main():
    video_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "tests", "video_decoder", "Видосик.mp4"
    )

    decoder = VideoDecoder(video_path)
    decoder.on_start()

    for frame in decoder.handle():
        print(frame.shape)

    decoder.on_exit()


if __name__ == "__main__":
    main()
