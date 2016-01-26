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

#TODO: improve docstrings

from PyMca5.PyMcaGui import PyMcaQt as qt
import os.path

QVERSION = qt.qVersion()

class LabelEntry(qt.QWidget):
    '''
    Composite widget with a label and a QLineEdit text field.
    
    :param label_text: Text displayed on the label on the left hand side.
                       If set to None, it defaults to "File filters"
    :type label_text: string or None
    :param entry_text: Predefined text in the QLineEdit widget.
    :type entry_text: string
    :param parent: Parent widget
    :type parent: QWidget or None
    '''
    #sigLabelEntry = qt.pyqtSignal(object)
     
    def __init__(self,label_text=None, entry_text='', parent=None):
        super(LabelEntry,self).__init__(parent)
        layout = qt.QHBoxLayout()
        
        if label_text is None:
            label_text = "File filters"
        self.label = qt.QLabel(label_text)
        layout.addWidget(self.label)
        
        self.lineEdit = qt.QLineEdit()
        self.lineEdit.setText(entry_text)
        layout.addWidget(self.lineEdit)
        
        self.setLayout(layout)
        
        # inherit from main QLineEdit signals, methods and attributes 
        self.textChanged = self.lineEdit.textChanged
        self.text = self.lineEdit.text
        self.setText = self.lineEdit.setText
        

class MyTreeView(qt.QTreeView):
    '''Regular QTreeView with an additional enterKeyPressed signal, 
    to pick files by pressing Enter or Return, and with
    column width auto-resizing.
    '''
    enterKeyPressed = qt.pyqtSignal()

    def __init__(self, parent = None, auto_resize=True):
        qt.QTreeView.__init__(self, parent)
        self._lastMouse = None
        if auto_resize:
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
        '''On mouse press events, remember which button was pressed
        in a self._lastMouse attribute.
        '''
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

                
class FileSystemTree(qt.QWidget):
    '''Composite widget with a QTreeView to display a file system tree
    and a QLineEdit text field in which users can specify a filter string, 
    to display only files whose names match specified wildcard patterns.

    Clicking or pressing Enter causes a sigFileSystemTree signal to be 
    emitted to broadcast a dictionary with information about the selected 
    file. 
    
    :param root_path: Root path for both the TreeView and FileSystemModel.
                      If unspecified or None, it will be set to the user 
                      home directory.
    :type root_path: string or None
    :param filter_strings: List of wildcard strings. When set, only files 
                           whose names match at least one filter are 
                           displayed.
    :type filter_strings: list of strings or None
    :param autosize_tree_columns: Flag to indicate if treeView columns are
                                  to be autosized dynamically 
    :type autosize_tree_columns: boolean default True
    :param hide_filter_entry: Flag to hide filter entry widget.
    :type hide_filter_entry: boolean default False
    '''  
    sigFileSystemTree = qt.pyqtSignal(object)
    
    def __init__(self, root_path=None, filter_strings=None, 
                 autosize_tree_columns=True, hide_filter_entry=False,
                 parent=None):
        self.autosize_tree_columns = autosize_tree_columns
        
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.filterEntry = LabelEntry()
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
        
        if root_path is None:
            root_path = os.path.expanduser('~')  
        self.setRootPath(root_path)

        self.filterEntry.textChanged.connect(self.setFilters)
        self.treeview.clicked.connect(self.itemClicked)
        self.treeview.doubleClicked.connect(self.itemDoubleClicked)
        self.treeview.enterKeyPressed.connect(self.itemEnterPressed)
               
    def setRootPath(self, root_path):
        '''Set the root path for the QFileSystemModel and the QTreeView.
        
        :param root_path: Root path. 
        :type root_path: string
        '''
        self.model.setRootPath(root_path)
        self.treeview.setRootIndex(self.model.index(root_path))

    def setFilters(self):
        '''Set filters for files to display in the treeview
        widget. The filters can be specified in a QLineEdit field as
        whitespace delimited wildcard strings.
        
        Example::
        
            *.py image[0-9][0-9][0-9].png *_???.pdf
        '''
        filter_strings = self.filterEntry.text().split()
        self.model.setNameFilters(filter_strings)
        self.model.setNameFilterDisables(False)
     
        if self.autosize_tree_columns:
            self.treeview.resizeAllColumns()
            
    def showFilterEntry(self):
        '''Show the text entry widget to enable users to define filters.'''
        self.filterEntry.show()
        
    def hideFilterEntry(self):
        '''Hide the text entry widget.'''
        self.filterEntry.hide()
        
    def itemClicked(self, modelIndex):
        '''
        :param modelIndex: Index within the QFileSystemModel of the clicked 
                           item.  
        :type modelIndex: QModelIndex
        '''
        event = "itemClicked"
        self.emitSignal(event, modelIndex)
        
    def itemDoubleClicked(self, modelIndex):
        '''
        :param modelIndex: Index within the QFileSystemModel of the 
                           double-clicked item.
        :type modelIndex: QModelIndex
        '''
        event = "itemDoubleClicked"
        self.emitSignal(event, modelIndex)
            
    def itemEnterPressed(self):
        '''
        :param modelIndex: Index within the QFileSystemModel of the item
                           selected when the Enter key was pressed.
        :type modelIndex: QModelIndex
        '''
        event = "itemEnterKeyPressed"
        modelIndex = self.treeview.selectedIndexes()[0]
        self.emitSignal(event, modelIndex)

    def emitSignal(self, event, modelIndex):
        '''Emits a sigFileSystemTree signal to broadcast a dictionary of 
        information about the selected item in the TreeView.
        
        :param modelIndex: Index within the QFileSystemModel of the item
                           selected when this method was called.
        :type modelIndex: QModelIndex
        :param event: Type of event that caused this method to be called: 
                      "itemEnterKeyPressed", "itemDoubleClicked" or 
                      "itemClicked"
        :type event: string
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
                                                 
        self.sigFileSystemTree.emit(ddict)

        
def test():
    import os.path, sys
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    
    try:
        root_path = sys.argv[1]
        if not os.path.isdir(root_path):
            raise IndexError
        filter_strings = sys.argv[2:]
        fftv = FileSystemTree(root_path, filter_strings)
    except IndexError:
        fftv = FileSystemTree()
    
    def mySlot(ddict):
        print(ddict)
        
    fftv.sigFileSystemTree.connect(mySlot)
    fftv.show()
    app.exec_()

if __name__ == '__main__':
    test()
