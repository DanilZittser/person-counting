from analytics.person_counting import PersonCountingAnalytics as Analytics
from handlers.handler import Handler


def main() -> None:
    decoder: Handler = VideoDecoder(video_filepath=...)  # todo
    decoder.on_start()

    analytics: Analytics = Analytics(...)
    analytics.on_start()

    for frame, frame_number, frame_timestamp in decoder.handle():
        print(f'Analytics event: {analytics.process_frame(image=frame)}')

    decoder.on_exit()
    analytics.on_exit()


if __name__ == '__main__':
    main()
