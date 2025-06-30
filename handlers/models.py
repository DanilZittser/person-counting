from dataclasses import dataclass

from nptyping import Float32, NDArray, Shape, UInt8


ImageRGB = NDArray[Shape['* height, * width, 3 rgb'], UInt8]
ImageBGR = NDArray[Shape['* height, * width, 3 bgr'], UInt8]
Image = ImageRGB | ImageBGR


BlobFloat = NDArray[Shape['*, *, *, *'], Float32]
BlobInt = NDArray[Shape['*, *, *, *'], UInt8]
Blob = BlobFloat | BlobInt


@dataclass
class Event:
    left_to_right: int
    right_to_left: int