from handlers.heuristic import VerticalLineHeuristic
from handlers.models import Box, Detection, Track


def create_track(track_id: int, start_x: int, end_x: int) -> Track:
    start = Detection(
        box=Box(left=start_x - 5, top=0, right=start_x + 5, bottom=10),
        score=1.0,
        label_as_int=0,
        label_as_str="person"
    )
    end = Detection(
        box=Box(left=end_x - 5, top=0, right=end_x + 5, bottom=10),
        score=1.0,
        label_as_int=0,
        label_as_str="person"
    )
    return Track(track_id=track_id, route=[start, end])


def test_vertical_line_heuristic_all_cases():
    heuristic = VerticalLineHeuristic(line_x=100)

    track_lr = create_track(track_id=1, start_x=50, end_x=150)   # слева направо
    track_rl = create_track(track_id=2, start_x=150, end_x=50)   # справа налево
    track_none = create_track(track_id=3, start_x=30, end_x=70)  # не пересёк

    events = heuristic.handle([track_lr, track_rl, track_none])

    # Проверка количества событий (только 2 из 3 треков пересекли линию)
    assert len(events) == 2

    # Преобразование списка событий в словарь по track_id для надёжной проверки
    events_by_id = {event.track_id: event for event in events}

    assert 1 in events_by_id
    assert events_by_id[1].left_to_right == 1
    assert events_by_id[1].right_to_left == 0

    assert 2 in events_by_id
    assert events_by_id[2].left_to_right == 0
    assert events_by_id[2].right_to_left == 1

    # Убедимся, что трек 3 не вернул событие
    assert 3 not in events_by_id
