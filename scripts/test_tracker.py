from handlers.tracker import IouTracker
from handlers.models import Box, Detection, Track


def moving_detection(
    frame_width: int = 340,
    box_width: int = 50,
    box_height: int = 100,
    step: int = 10,
    score: float = 0.9
):
    x1 = 0
    y1 = 100
    while x1 + box_width <= frame_width:
        box = Box(left=x1, top=y1, right=x1 + box_width, bottom=y1 + box_height)
        det = Detection(
            box=box,
            score=score,
            label_as_int=1,
            label_as_str="person"
        )
        yield [det]
        x1 += step


def main():
    tracker = IouTracker(iou_threshold=0.5, min_input_score=0.3, score_threshold=0.6, max_missed=1)
    tracker.on_start()

    gen = moving_detection()
    finished_tracks: list[Track] = []

    # Обработка кадров с объектом
    for detections in gen:
        finished = tracker.handle(detections)
        finished_tracks.extend(finished)

    # Добавление кадров без объекта для завершения трека
    for missed_frame in range(2):
        finished = tracker.handle([])
        finished_tracks.extend(finished)

    tracker.on_exit()

    # Проверки
    track = finished_tracks[0]
    assert len(finished_tracks) == 1
    assert isinstance(track, Track)
    assert track.track_id == 1
    assert len(track.route) > 1


if __name__ == "__main__":
    main()
