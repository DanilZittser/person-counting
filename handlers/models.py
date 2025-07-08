from dataclasses import dataclass
from typing import List

from nptyping import Float32, NDArray, Shape, UInt8

# Типы изображений
ImageRGB = NDArray[Shape['* height, * width, 3 rgb'], UInt8]
ImageBGR = NDArray[Shape['* height, * width, 3 bgr'], UInt8]
Image = ImageRGB | ImageBGR

# Типы blob-данных
BlobFloat = NDArray[Shape['*, *, *, *'], Float32]
BlobInt = NDArray[Shape['*, *, *, *'], UInt8]
Blob = BlobFloat | BlobInt


@dataclass
class Box:
    """Класс для хранения координат ограничивающего прямоугольника обнаруженного объекта."""
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class Detection:
    """Класс для хранения информации об обнаруженных объектах."""
    box: Box
    score: float
    label_as_int: int
    label_as_str: str


@dataclass
class Detections:
    """Класс-обёртка над списком обнаружений для совместимости с пайплайном."""
    detections: List[Detection]


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

@dataclass
class Events:
    Events: list[Event]

