from typing import Optional

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
import numpy


def size_transform(src: QtCore.QSize, dst: QtCore.QSize) -> QtGui.QTransform:
    """Compute the transformation matrix from one size to another."""
    sx = dst.width() / src.width()
    sy = dst.height() / src.height()
    return QtGui.QTransform().scale(sx, sy)


class ImageView(QtWidgets.QWidget):
    """Fixed-sized widget for displaying a dynamically mutated QImage."""
    image: QtGui.QImage
    _size_hint: QtCore.QSize
    _paint_transform: QtGui.QTransform

    def __init__(self, image: QtGui.QImage, size: Optional[QtCore.QSize] = None):
        super().__init__()
        self.image = image
        self._size_hint = size or image.size()

    def sizeHint(self) -> QtCore.QSize:
        return self._size_hint

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self._paint_transform = size_transform(self.size(), self.image.size())

    def paintEvent(self, event: QtGui.QPaintEvent):
        with QtGui.QPainter(self) as painter:
            dst = event.rect().toRectF()
            src = self._paint_transform.mapRect(dst)
            painter.drawImage(dst, self.image, src)


class ImageArray:
    """Collection of numpy array views into a QImage."""
    image: QtGui.QImage
    xy: numpy.ndarray
    bgra: numpy.ndarray
    rgb: numpy.ndarray
    red: numpy.ndarray
    green: numpy.ndarray
    blue: numpy.ndarray

    def __init__(self, image: QtGui.QImage):
        assert image.format() == QtGui.QImage.Format_RGB32
        shape = (image.size().width(), image.size().height(), 4)
        self.image = image
        self.xy = numpy.indices(shape[:2])[::-1]  # column-major
        self.bgra = numpy.ndarray(shape, dtype=numpy.uint8, buffer=image.bits())
        self.rgb = self.bgra[:,:,2::-1]
        self.red = self.bgra[:,:,2]
        self.green = self.bgra[:,:,1]
        self.blue = self.bgra[:,:,0]


def main(argv: list[str]) -> int:
    app = QtWidgets.QApplication(argv)
    window = QtWidgets.QMainWindow()
    window.setWindowTitle('genart')

    image = QtGui.QImage(QtCore.QSize(256, 256), QtGui.QImage.Format_RGB32)
    image.fill('magenta')

    # red = x, green = y, blue = checker
    array = ImageArray(image)
    rg = numpy.moveaxis(array.xy, 0, -1)
    checker = -(numpy.sum(array.xy, 0) % 2) & 0xff
    numpy.copyto(array.rgb, numpy.dstack((rg, checker)), casting='unsafe')

    view = ImageView(image, QtCore.QSize(512, 512))
    window.setCentralWidget(view)
    window.show()
    return app.exec()