from handlers.handler import Handler
from handlers.models import Detection, Track, Box
from typing import List


def iou(box1: Box, box2: Box) -> float:
    x1 = max(box1.left, box2.left)
    y1 = max(box1.top, box2.top)
    x2 = min(box1.right, box2.right)
    y2 = min(box1.bottom, box2.bottom)

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1.right - box1.left + 1) * (box1.bottom - box1.top + 1)
    box2_area = (box2.right - box2.left + 1) * (box2.bottom - box2.top + 1)

    return inter_area / float(box1_area + box2_area - inter_area)


class IouTracker(Handler):
    def __init__(self, iou_threshold: float = 0.5, min_input_score: float = 0.3, score_threshold: float = 0.6,
                 max_missed: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_input_score = min_input_score
        self.score_threshold = score_threshold
        self.next_id = 1
        self.active_tracks: dict[int, dict] = {}
        self.finished_tracks: list[Track] = []
        self.frame_idx = 0

    def on_start(self, *args, **kwargs):
        self.active_tracks.clear()
        self.finished_tracks.clear()
        self.next_id = 1
        self.frame_idx = 0

    def handle(self, detections: List[Detection]) -> List[Track]:
        self.frame_idx += 1
        filtered_detections = [d for d in detections if d.score >= self.min_input_score]
        unmatched_detections = filtered_detections.copy()

        # Сопоставление детекций с активными треками
        for track_id, track_data in list(self.active_tracks.items()):
            best_match = None
            best_iou = 0.0

            for det in unmatched_detections:
                if det.label_as_int != track_data['route'][-1].label_as_int:
                    continue
                iou_score = iou(det.box, track_data['route'][-1].box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_match = det

            if best_iou >= self.iou_threshold and best_match:
                track_data['route'].append(best_match)
                track_data['last_frame'] = self.frame_idx
                unmatched_detections.remove(best_match)

        # Завершить треки объектов, которые не проходят по порогу пропущеных кадров(max_missed)
        for track_id in list(self.active_tracks):
            if self.frame_idx - self.active_tracks[track_id]['last_frame'] > self.max_missed:
                route = self.active_tracks[track_id]['route']
                if any(det.score >= self.score_threshold for det in route):
                    self.finished_tracks.append(Track(track_id=track_id, route=route))
                del self.active_tracks[track_id]

        # Создание новых треков для не сопоставленных детекций
        for det in unmatched_detections:
            self.active_tracks[self.next_id] = {
                'route': [det],
                'last_frame': self.frame_idx
            }
            self.next_id += 1

        # Вернуть завершённые треки и очистить список
        completed_tracks = self.finished_tracks
        self.finished_tracks = []
        return completed_tracks

    def on_exit(self, *args, **kwargs):
        self.active_tracks.clear()
        self.finished_tracks.clear()
