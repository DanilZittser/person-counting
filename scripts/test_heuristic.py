import sys
import os

# Добавляем корень проекта в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from handlers.heuristic import VerticalLineHeuristic
from handlers.models import Box, Detection, Track


def create_test_track(track_id: int, start_x: int, end_x: int) -> Track:
    """Создаёт Track с двумя Detection: начальной и конечной позицией по x."""
    detection_start = Detection(
        box=Box(left=start_x - 10, top=0, right=start_x + 10, bottom=20),
        score=0.9,
        label_as_int=0,
        label_as_str="person"
    )

    detection_end = Detection(
        box=Box(left=end_x - 10, top=0, right=end_x + 10, bottom=20),
        score=0.95,
        label_as_int=0,
        label_as_str="person"
    )

    return Track(track_id=track_id, route=[detection_start, detection_end])


def main():
    # Порог вертикальной линии
    line_x = 100
    heuristic = VerticalLineHeuristic(line_x=line_x)

    # Треки с разными направлениями
    track_lr = create_test_track(track_id=1, start_x=50, end_x=150)    # слева направо
    track_rl = create_test_track(track_id=2, start_x=150, end_x=50)    # справа налево
    track_no_cross = create_test_track(track_id=3, start_x=30, end_x=70)  # не пересёк

    tracks = [track_lr, track_rl, track_no_cross]
    events = heuristic.handle(tracks)

    for event in events:
        print(f"Track ID: {event.track_id}, "
              f"Left to Right: {event.left_to_right}, "
              f"Right to Left: {event.right_to_left}")


if __name__ == '__main__':
    main()
