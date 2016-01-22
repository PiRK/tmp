import sys
from PyMca5.PyMcaGui import PyMcaQt as qt

class MyListWidget(qt.QListWidget):
  def __init__(self, parent):
    super(MyListWidget, self).__init__(parent)
    self.setAcceptDrops(True)
    self.setDragDropMode(qt.QAbstractItemView.InternalMove)

  def dragEnterEvent(self, event):
    if event.mimeData().hasUrls() or event.mimeData().hasText():
      event.acceptProposedAction()
    else:
      super(MyListWidget, self).dragEnterEvent(event)

  def dropEvent(self, event):
    if event.mimeData().hasUrls():
      for url in event.mimeData().urls():
        self.addItem(url.path())
      event.acceptProposedAction()
    elif event.mimeData().hasText():
      self.addItem(event.mimeData().text())
    else:
      super(MyListWidget,self).dropEvent(event)

class MyWindow(qt.QWidget):
  def __init__(self):
    super(MyWindow,self).__init__()
    self.setGeometry(100,100,300,400)
    self.setWindowTitle("Filenames")

    self.list = MyListWidget(self)
    layout = qt.QVBoxLayout(self)
    layout.addWidget(self.list)

    self.setLayout(layout)

if __name__ == '__main__':

  app = qt.QApplication(sys.argv)
  app.setStyle("plastique")

  window = MyWindow()
  window.show()

  sys.exit(app.exec_())
