from handlers.handler import Handler
from handlers.models import Track, Event


class VerticalLineHeuristic(Handler):
    """
    Компонент VAP: определение направления пересечения виртуальной вертикальной линии.
    """

    def __init__(self, line_x: int):
        """
        :param line_x: x-координата вертикальной линии
        """
        self.line_x = line_x

    def handle(self, tracks: list[Track]) -> list[Event]:
        events = []

        for track in tracks:
            if len(track.route) < 2:
                continue  # не хватает данных

            start_det = track.route[0]
            end_det = track.route[-1]

            x_start = (start_det.box.left + start_det.box.right) / 2
            x_end = (end_det.box.left + end_det.box.right) / 2

            if x_start < self.line_x < x_end:
                events.append(Event(track_id=track.track_id, left_to_right=1, right_to_left=0))
            elif x_start > self.line_x > x_end:
                events.append(Event(track_id=track.track_id, left_to_right=0, right_to_left=1))

        return events
