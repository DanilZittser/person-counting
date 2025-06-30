from dataclasses import dataclass

from nptyping import Float32, NDArray, Shape, UInt8


ImageRGB = NDArray[Shape['* height, * width, 3 rgb'], UInt8]
ImageBGR = NDArray[Shape['* height, * width, 3 bgr'], UInt8]
Image = ImageRGB | ImageBGR


BlobFloat = NDArray[Shape['*, *, *, *'], Float32]
BlobInt = NDArray[Shape['*, *, *, *'], UInt8]
Blob = BlobFloat | BlobInt


@dataclass
class Box:
    """Класс для хранения координат ограничивающего прямоугольника обнаруженного объекта."""
    left: int  # x-координата левого верхнего угла прямоугольной ограничивающей рамки
    top: int  # y-координата левого верхнего угла
    right: int  # y-координата правого нижнего угла
    bottom: int  # y-координата правого нижнего угла


@dataclass
class Detection:
    """Класс для хранения информации об обнаруженных объектах."""
    box: Box
    score: float
    label_as_int: int
    label_as_str: str


Detections: list[Detection]  # все обнаружения объектов в одном кадре


@dataclass
class Track:
    track_id: int  # идентификатор трека
    route: list[Detection]  # список обнаружений ОДНОГО И ТОГО ЖЕ объекта на последовательности кадров


Tracks: list[Track]


@dataclass
class Event:
    track_id: int
    left_to_right: int
    right_to_left: int


Events: list[Event]
