from typing import Optional

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
import numpy

from genart import demo
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
        try:
            result = p.run(self.array.xy)
            [dy, dx] = numpy.gradient(result)
            theta = numpy.angle(dx + dy * 1j)
            numpy.copyto(self.array.blue, result % 256, casting='unsafe')
            numpy.copyto(self.array.red, result // 256, casting='unsafe')
            numpy.copyto(self.array.green, 128 * (1 + numpy.cos(theta)), casting='unsafe')
        except program.ExecutionError:
            self.image.fill('magenta')
        self.view.update()


class SaveDialog(QtWidgets.QFileDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFileMode(self.FileMode.AnyFile)
        self.setAcceptMode(self.AcceptMode.AcceptSave)
        self.setMimeTypeFilters(['image/png'])
        self.setDefaultSuffix('png')


class ProgramItem(QtWidgets.QListWidgetItem):
    program: program.Program

    def __init__(self, p: program.Program):
        super().__init__()
        self.program = p
        self.setText(str(p))


class UI:
    app: QtWidgets.QApplication
    window: QtWidgets.QMainWindow
    save_dialog: SaveDialog
    art: RandomArt
    programs: QtWidgets.QListWidget

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__()
        self.app = app
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle('genart')
        self.save_dialog = SaveDialog(self.window)
        self.save_dialog.fileSelected.connect(self.save_program)
        self.art = RandomArt()
        self.programs = QtWidgets.QListWidget()
        self.programs.currentItemChanged.connect(self.select_program)

        center = QtWidgets.QWidget()
        self.window.setCentralWidget(center)

        hlayout = QtWidgets.QHBoxLayout(center)
        vlayout = QtWidgets.QVBoxLayout()
        hlayout.addLayout(vlayout)
        hlayout.addWidget(self.art.view)
        vlayout.addWidget(self.programs)

        buttons = QtWidgets.QGridLayout()
        vlayout.addLayout(buttons)

        copy = QtWidgets.QPushButton('Copy')
        copy.clicked.connect(self.copy_program)
        buttons.addWidget(copy, 0, 0)

        save = QtWidgets.QPushButton('Save')
        save.clicked.connect(self.save_dialog.open)
        buttons.addWidget(save, 0, 1)

        reroll = QtWidgets.QPushButton('Reroll')
        reroll.clicked.connect(self.reroll)
        buttons.addWidget(reroll, 1, 0, 1, 2)

        for d in demo.DEMOS:
            self.programs.addItem(ProgramItem(d))
        self.programs.setCurrentRow(0)

    def select_program(self, new: ProgramItem, old: ProgramItem):
        self.art.render(new.program)

    def copy_program(self):
        item = self.programs.currentItem()
        clipboard = self.app.clipboard()
        clipboard.setText(str(item.program))

    def save_program(self, filename: str):
        item = self.programs.currentItem()
        image = self.art.image.convertedToColorSpace(QtGui.QColorSpace.SRgb)
        image.setText('Software', 'sjolsen/genart')
        image.setText('Comment', str(item.program))
        image.save(filename)

    def reroll(self):
        item = ProgramItem(program.generate_program())
        self.programs.addItem(item)
        self.programs.setCurrentItem(item)


def main(argv: list[str]) -> int:
    app = QtWidgets.QApplication(argv)
    ui = UI(app)
    ui.window.show()
    return app.exec()
