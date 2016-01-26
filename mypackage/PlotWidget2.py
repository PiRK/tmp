#!/usr/local/bin/python
# -*- coding: utf-8 -*-
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
This module provides an overloaded :class:PlotWidget supporting drag and drop. 
"""

#TODO: don't subclass PlotWidget, just include it in a QWidget that takes care 
#      of our dropEvent

import csv
import sys

from PyMca5.PyMcaGui.plotting.PlotWidget import PlotWidget
from PyMca5.PyMcaGui import PyMcaQt as qt

class CSVData():
    '''This class parses CSV data files.'''
    def __init__(self, csvpath):
        '''      
        :param csvpath: Path of a CSV file
        :type csvpath: string
        '''
        self.csvpath = csvpath
        self.parse()
        
    def parse(self): 
        '''
        Parse CSV file to populate *self.hdrs* (list of strings) and 
        *self.data* (list of list of values).
        '''   
        with open(self.csvpath) as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            has_header = csv.Sniffer().has_header(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect)
            if has_header:
                self.hdrs = next(reader)
            self.data_rows = []
            for row in reader:
                self.data_rows.append([float(value) for value in row])
                
    def get_ycols(self, columns=None):
        '''
        Returns a list of lists, each containing data for a column.
        
        :param columns: List of indexes of data columns to be retrieved. 
                        The first column has index 0.
        :type columns: list of int or None
        :returns: List of lists of values, one list per column
        :rtype: list of lists of floats
        '''
        ycols = []
        if columns is None:
            # No column indexes specified, read all columns except 1st
            columns = range(1, len(self.data_rows[0]))
        for i in columns:
            ycols.append([])
        for row in self.data_rows:
            for i in columns:
                ycols[columns.index(i)].append(row[i])
                     
        return ycols    
    
    def get_xcol(self, column_index=0):
        '''
        Returns a single list of values for one specified column.
        :param column_index: Index of the column to be retrieved.
        :type column_index: int
        :returns: List of values from a column
        :rtype: list of floats
        '''   
        xcol = []
        for row in self.data_rows:
            xcol.append(row[column_index])                    
        return xcol
                
    def get_hdrs(self, columns=None):
        '''
        :param columns: List of header indexes to be retrieved.
        :type columns: list of integers 
        :returns: List of headers
        :rtype: list of strings
        '''
        if columns is None:
            # No column indexes specified, read all columns 
            columns = range(len(self.hdrs))
        return [self.hdrs[i] for i in columns]
                
    def get_hdr(self, column_index=0):
        '''
        :param column_index: index of header to be retrieved.
        :type column_index: int 
        :returns: Header
        :rtype: string
        '''
        return self.hdrs[column_index]

class PlotWidget2(qt.QWidget):
    """This widget overloads PlotWidget with methods enabling drag and drop 
    of data files into this widget. At the moment, only CSV files are supported.
    The first column will be assumed to be data for the x axis. All other columns
    will be plotted against the first one.
    """
    def __init__(self, parent=None, backend=None,
                 legends=False, callback=None, **kw):
        qt.QWidget.__init__(self, parent)
        
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.plotwidget = PlotWidget(self, backend,
                                legends, callback, **kw)
        layout.addWidget(self.plotwidget)
        
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()

    def dropEvent(self, e):
        data_file_path = None
        if e.mimeData().hasUrls():
            # we accept only single files
            if e.mimeData().urls()[0].path().lower().endswith('.csv'):
                data_file_path = e.mimeData().urls()[0].path()
        if data_file_path is not None:
            if sys.platform.startswith("win") and \
               data_file_path.startswith("/") and \
               ":/" in data_file_path:
                data_file_path = data_file_path[1:]              
            data = CSVData(data_file_path)
            x = data.get_xcol(0)
            ys = data.get_ycols()
            hdrs = data.get_hdrs()
            (xhdr, yhdrs) = (hdrs[0], hdrs[1:])
            for (y, yhdr) in zip(ys, yhdrs):
                self.plotwidget.addCurve(x, y, legend=yhdr, xlabel=xhdr, ylabel=yhdr)
                
    def keyPressEvent(self, event):
        if (event.modifiers() & qt.Qt.ShiftModifier) and (event.modifiers() & qt.Qt.ControlModifier):
            if event.key() == qt.Qt.Key_C:
                #print("Shift + Ctrl + C pressed")
                self.renderToClipboard()
        qt.QWidget.keyPressEvent(self, event)
                
    def renderToClipboard(self):
        pixmap = qt.QPixmap(self.size())
        self.render(pixmap)
        qt.QApplication.clipboard().setPixmap(pixmap)
        
            
if __name__ == "__main__":
    import time
    import sys
    from PyMca5.PyMcaGui import PyMcaQt as qt
    backend = None
    if ("matplotlib" in sys.argv) or ("mpl" in sys.argv):
        backend = "matplotlib"
        print("USING matplotlib")
        time.sleep(1)
    elif ("pyqtgraph" in sys.argv):
        backend = "pyqtgraph"
        print("USING PyQtGraph")
        time.sleep(1)
    elif ("OpenGL" in sys.argv) or ("opengl" in sys.argv) or ("gl" in sys.argv):
        backend = "opengl"
        print("USING OpenGL")
        time.sleep(1)  
    elif ("GLUT" in sys.argv) or ("glut" in sys.argv):
        backend = "glut"
        print("USING GLUT")
        time.sleep(1)
    else:
        print ("USING default backend")
        time.sleep(1)
    app = qt.QApplication([])
    plot = PlotWidget2(None, backend=backend, legends=True)
    plot.plotwidget.setPanWithArrowKeys(True)
    plot.show()
    app.exec_()

        