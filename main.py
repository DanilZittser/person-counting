from handlers.decoder import VideoDecoder
from handlers.pre_processor import PreProcessor
from handlers.inference import YoloInference
from handlers.post_processor import PostProcessor
from handlers.tracker import IouTracker
from handlers.heuristic import VerticalLineHeuristic
from analytics.person_counting import PersonCountingAnalytics as Analytics


def main():
    video_path = "assets/video1.mp4" # Путь к видео

    decoder = VideoDecoder(video_path)
    pre_processor = PreProcessor()
    inference = YoloInference(model_path="yolo11n.pt", conf_threshold=0.5)
    post_processor = PostProcessor()
    tracker = IouTracker(iou_threshold=0.5, min_input_score=0.3, score_threshold=0.6,  max_missed=5)
    heuristic = VerticalLineHeuristic(line_x=427) # line_x = frame_width // 2

    analytics = Analytics(
        pre_processor=pre_processor,
        inference=inference,
        post_processor=post_processor,
        tracker=tracker,
        heuristic=heuristic,
    )

    decoder.on_start()
    analytics.on_start()
    counter = 1

    for frame in decoder.handle():
        events = analytics.process_frame(frame)
        for event in events.Events:
            print(f"{counter}. Человек {event.track_id} пересёк линию: "
                  f"LeftToRight={event.left_to_right}, RightToLeft={event.right_to_left}")
            counter += 1


    decoder.on_exit()
    analytics.on_exit()


if __name__ == '__main__':
    main()
