from typing import Optional

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
import numpy

from genart import program


def size_transform(src: QtCore.QSize, dst: QtCore.QSize) -> QtGui.QTransform:
    """Compute the transformation matrix from one size to another."""
    sx = dst.width() / src.width()
    sy = dst.height() / src.height()
    return QtGui.QTransform().scale(sx, sy)


class ImageView(QtWidgets.QWidget):
    """Fixed-sized widget for displaying a dynamically mutated QImage."""
    image: QtGui.QImage
    resample: bool
    _size_hint: QtCore.QSize
    _paint_transform: QtGui.QTransform
    _paint_buffer: QtGui.QImage

    def __init__(self, image: QtGui.QImage, *,
                 size: Optional[QtCore.QSize] = None,
                 resample: bool = False):
        super().__init__()
        self.image = image
        self.resample = resample
        self._size_hint = size or image.size()

    def sizeHint(self) -> QtCore.QSize:
        return self._size_hint

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self._paint_transform = size_transform(self.size(), self.image.size())
        self._paint_buffer = QtGui.QImage(self.size(), self.image.format())

    def paintEvent(self, event: QtGui.QPaintEvent):
        # Perform the scaling in the image's native color space
        self._paint_buffer.setColorSpace(self.image.colorSpace())
        with QtGui.QPainter(self._paint_buffer) as painter:
            painter.setRenderHint(
                QtGui.QPainter.SmoothPixmapTransform, self.resample)
            dst = event.rect().toRectF()
            src = self._paint_transform.mapRect(dst)
            painter.drawImage(dst, self.image, src)

        # Paint the scaled image in non-linear sRGB
        self._paint_buffer.convertToColorSpace(QtGui.QColorSpace.SRgb)
        with QtGui.QPainter(self) as painter:
            painter.drawImage(event.rect(), self._paint_buffer, event.rect())


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
        self.bgra = numpy.ndarray(
            shape, dtype=numpy.uint8, buffer=image.bits())
        self.rgb = self.bgra[:, :, 2::-1]
        self.red = self.bgra[:, :, 2]
        self.green = self.bgra[:, :, 1]
        self.blue = self.bgra[:, :, 0]


class RandomArt:
    image: QtGui.QImage
    view: ImageView
    array: ImageArray

    def __init__(self):
        super().__init__()
        size = QtCore.QSize(256, 256)
        self.image = QtGui.QImage(size, QtGui.QImage.Format_RGB32)
        self.image.setColorSpace(QtGui.QColorSpace.SRgbLinear)
        self.image.fill('magenta')
        self.view = ImageView(self.image, size=QtCore.QSize(512, 512))
        self.array = ImageArray(self.image)

    def render(self, p: program.Program):
        print(str(p))
        result = p.run(self.array.xy) % 256
        numpy.copyto(self.array.rgb,
                     result[:, :, numpy.newaxis], casting='unsafe')
        self.view.update()

    def reroll(self):
        self.render(program.random_program())


def main(argv: list[str]) -> int:
    app = QtWidgets.QApplication(argv)
    window = QtWidgets.QMainWindow()
    window.setWindowTitle('genart')

    art = RandomArt()
    art.render(program.demo())

    button = QtWidgets.QPushButton('Reroll')
    button.clicked.connect(art.reroll)

    center = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(center)
    layout.addWidget(art.view)
    layout.addWidget(button)

    window.setCentralWidget(center)
    window.show()
    return app.exec()
