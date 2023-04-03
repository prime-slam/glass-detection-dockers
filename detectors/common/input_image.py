from pathlib import Path


class InputImage:
    def __init__(self, img, path: Path, w: int, h: int):
        self.img = img
        self.path = path
        self.w = w
        self.h = h
