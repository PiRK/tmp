#/*##########################################################################
# Copyright (C) 2004-2015 European Synchrotron Radiation Facility, Grenoble, France
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "P. Knobel - ESRF Data Analysis"
__contact__ = "pierre.knobel@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This widget is a TreeView specialized in displaying the file system. 
A filter can be set as a string with wildcards, to show only files whose names 
match a specified pattern. 
"""

from PyMca5.PyMcaGui import PyMcaQt as qt
QVERSION = qt.qVersion()

class FilterEntry(qt.QWidget):
    '''
    '''
    #sigFilterEntry = qt.pyqtSignal(object)
     
    def __init__(self,label=None, filter_text=None, parent=None):
        super(FilterEntry,self).__init__(parent)
        layout = qt.QHBoxLayout()
        
        if label is None:
            label = "File filters (space delimited with wildcards)"
        self.label = qt.QLabel(label)
        layout.addWidget(self.label)
        
        self.lineEdit = qt.QLineEdit()
        self.lineEdit.setText(filter_text)
        layout.addWidget(self.lineEdit)
        
        self.setLayout(layout)
        
        #self.lineEdit.textChanged.connect(self.textChangedEvent)
        self.textChanged = self.lineEdit.textChanged
        self.text = self.lineEdit.text
        
    def setText(self, text):
        self.lineEdit.setText(text)
        
#     def textChangedEvent(self, e):
#         filters = self.lineEdit.text().split()
#         ddict = {'event': 'textChanged',
#                  'filters': filters}
#         self.sigFilterEntry.emit(ddict)


class MyTreeView(qt.QTreeView):
    '''Regular QTreeView with an additional enterKeyPressed signal, 
    to pick files by pressing Enter or Return, and with
    column width auto-resizing.
    '''
    enterKeyPressed = qt.pyqtSignal()

    def __init__(self, parent = None):
        qt.QTreeView.__init__(self, parent)
        self._lastMouse = None
        self.expanded.connect(self.resizeAllColumns)
        self.collapsed.connect(self.resizeAllColumns)
        
    def resizeAllColumns(self):
        for i in range(0, self.model().columnCount()):
            self.resizeColumnToContents(i)  

    def keyPressEvent(self, event):
        if event.key() in [qt.Qt.Key_Enter, qt.Qt.Key_Return]:
            self.enterKeyPressed.emit()
        qt.QTreeView.keyPressEvent(self, event)

    def mousePressEvent(self, e):
        button = e.button()
        if button == qt.Qt.LeftButton:
            self._lastMouse = "left"
        elif button == qt.Qt.RightButton:
            self._lastMouse = "right"
        elif button == qt.Qt.MidButton:
            self._lastMouse = "middle"
        else:
            #Should I set it to no button?
            self._lastMouse = "left"
        qt.QTreeView.mousePressEvent(self, e)
        if self._lastMouse != "left":
            # Qt5 only sends itemClicked on left button mouse click
            if QVERSION > "5":  
                event = "itemClicked"
                modelIndex = self.indexAt(e.pos())
                self.emitSignal(event, modelIndex)

                
class FileSystemWidget(qt.QWidget):
    '''Composite widget with a QTreeView to display a file system tree
    and a QLineEdit text field in which users can specify a filter string, 
    to display only files whose names match specified wildcard patterns.

    Clicking or pressing Enter causes a sigFileSystemWidget signal to be 
    emitted to broadcast a dictionary with information about the selected 
    file. 
    '''  
    sigFileSystemWidget = qt.pyqtSignal(object)
    
    def __init__(self, root_path, filter_strings=None, 
                 autosize_tree_columns=True, hide_filter_entry=False,
                 parent=None):
        '''
        :param root_path: Root path for both the TreeView and FileSystemModel
        :type root_path: string
        :param filter_strings: List of wildcard strings. When set, only files 
                               whose names match at least one filter are 
                               displayed.
        :type filter_strings: list of strings default None
        :param autosize_tree_columns: Flag to indicate if treeView columns are
                                      to be autosized dynamically 
        :type autosize_tree_columns: boolean default True
        :param hide_filter_entry: Flag to hide filter entry widget.
        :type hide_filter_entry: boolean default False
        '''
        self.autosize_tree_columns = autosize_tree_columns
        
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.filterEntry = FilterEntry()
        if not filter_strings is None:
            self.filterEntry.setText(' '.join(filter_strings))
        layout.addWidget(self.filterEntry)
        if hide_filter_entry:
            self.hideFilterEntry()
          
        self.model = qt.QFileSystemModel()
        self.treeview = MyTreeView(self) 
        self.treeview.setSortingEnabled(True)
        self.treeview.setModel(self.model)
        self.treeview.setDragEnabled(True)
        layout.addWidget(self.treeview)

        self.setFilters()
        self.setRootPath(root_path)

        self.filterEntry.textChanged.connect(self.setFilters)
        self.treeview.clicked.connect(self.itemClicked)
        self.treeview.doubleClicked.connect(self.itemDoubleClicked)
        self.treeview.enterKeyPressed.connect(self.itemEnterPressed)
       
    def setRootPath(self, root_path):
        self.model.setRootPath(root_path)
        self.treeview.setRootIndex(self.model.index(root_path))

    def setFilters(self):
        '''Set filters for files to display in the treeview
        widget. The filters can be specified in a text field as
        whitespace delimited wildcard strings.
        Example: 
            *.py image[0-9][0-9][0-9].png *_???.pdf
        '''
        filter_strings = self.filterEntry.text().split()
        self.model.setNameFilters(filter_strings)
        self.model.setNameFilterDisables(False)
     
        if self.autosize_tree_columns:
            self.treeview.resizeAllColumns()
            
    def showFilterEntry(self):
        self.filterEntry.show()
        
    def hideFilterEntry(self):
        self.filterEntry.hide()
        
    def itemClicked(self, modelIndex):
        '''
        '''
        event = "itemClicked"
        self.emitSignal(event, modelIndex)
        
    def itemDoubleClicked(self, modelIndex):
        '''
        '''
        event = "itemDoubleClicked"
        self.emitSignal(event, modelIndex)
            
    def itemEnterPressed(self):
        '''
        '''
        event = "itemEnterKeyPressed"
        modelIndex = self.treeview.selectedIndexes()[0]
        self.emitSignal(event, modelIndex)

    def emitSignal(self, event, modelIndex):
        '''
        '''
        fileInfo = self.model.fileInfo(modelIndex)
        ddict = {}
        ddict['event'] = event
        ddict['name'] = fileInfo.fileName()
        ddict['path'] = fileInfo.absoluteFilePath()
        ddict['basename'] = fileInfo.baseName()                  # (a.tar.gz -> a)
        ddict['completebasename'] = fileInfo.completeBaseName()  # ( -> a.tar)
        ddict['completesuffix'] = fileInfo.completeSuffix()      # ( -> tar.gz)
        ddict['suffix'] = fileInfo.suffix()                      # ( -> gz)
        ddict['size'] = fileInfo.size()                          # in bytes
        ddict['modified iso'] = fileInfo.lastModified().toString(
                                                             qt.Qt.ISODate)
        
        if not "Clicked" in event: 
            ddict['mouse'] = 'None'
        else:
            ddict['mouse'] = self.treeview._lastMouse * 1
            
        # retrieve file type and date as displayed in the widget 
        this_row = modelIndex.row()
        ddict['type'] = self.model.data(modelIndex.sibling(this_row, 2))
        ddict['modified'] = self.model.data(modelIndex.sibling(this_row, 3))
                                                 
        self.sigFileSystemWidget.emit(ddict)

        
def test():
    import os.path, sys
    try:
        root_path = sys.argv[1]
        if not os.path.isdir(root_path):
            raise IndexError
        filter_strings = sys.argv[2:]
    except IndexError:
        root_path  = qt.QDir.currentPath()
        filter_strings = ['*.py', '*.pdf']
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    fftv = FileSystemWidget(root_path, filter_strings)
    def mySlot(ddict):
        print(ddict)
    fftv.sigFileSystemWidget.connect(mySlot)
    fftv.show()
    app.exec_()

if __name__ == '__main__':
    test()
