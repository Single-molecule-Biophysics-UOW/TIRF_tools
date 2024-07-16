# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:57:37 2024

@author: smueller
"""

try:
    import napari
    from qtpy import QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvas as Canvas, NavigationToolbar2QT as NavigationToolbar
except ImportError:
    raise ImportError(
        "Napari is required to use visualization tools. "
        "Install with `pip install -i https://test.pypi.org/simple/ tirf-tools[visualize]` to install all dependencies"
    )

from typing import TYPE_CHECKING
import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget, QComboBox, QLabel, QSpinBox, QCheckBox
from matplotlib.figure import Figure
from napari.layers import Points
if TYPE_CHECKING:
    import napari
        
class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent=parent)

        self.figure = Figure(tight_layout=True)
        
        self.figure.patch.set_facecolor('none')
        
        self.canvas = Canvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(self.toolbar)
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.line1 = self.format_axes(self.ax,'white')
        self.line2 = self.format_axes(self.ax2,'gold')
        
        #create dummy lines
        # self.line1, = self.ax.plot([0,0],[0,0],color='white')
        # self.line2, = self.ax2.plot([0,0],[0,0],color='red')
        #create dummy scatter
        self.scatter = self.ax.scatter(None,None,color='yellow',marker='x')
        # self.line1 = Line2D([0,0],[0,0], animated = True)#self.ax.plot()
        # self.ax.add_line(self.line1)
        
    def format_axes(self,ax, color):
        ax.patch.set_facecolor('none')
        ax.xaxis.label.set_color(color)        #setting up X-axis label color to yellow
        ax.yaxis.label.set_color(color)          #setting up Y-axis label color to blue
        ax.tick_params(axis='x', colors=color)    #setting up X-axis tick color to red
        ax.tick_params(axis='y', colors=color)  #setting up Y-axis tick color to black
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['bottom'].set_color('white')
        line, = ax.plot([0,0],[0,0],color=color)
        return line
        
        
    def update_limits(self):
        y1 = self.line1.get_ydata()
        y2 = self.line2.get_ydata()
        ymax = max(np.max(y1),np.max(y2))
        ymin = min(np.min(y1),np.min(y2))
        x1 = self.line1.get_xdata()
        
        xmax = np.max(x1)
        xmin = np.min(x1)
        
        
        if ymin < ymax:    
            self.ax.set_ylim([ymin,ymax])
            self.ax2.set_ylim([ymin,ymax])
        if xmin<xmax:
            self.ax.set_xlim([xmin,xmax])
    def update(self,line, xdata, ydata):        
        try:
            # line.set_ydata(xdata)
            line.set_data(xdata,ydata)
            self.update_limits()
            
        except:
            print('invalid data')
        self.update_limits()
        # self.ax.set_xlim([np.min(xdata),np.max(xdata)])
        # self.ax.set_ylim([np.min(ydata),np.max(ydata)])
        self.canvas.draw()
    def update_scatter(self, x,y):
        self.scatter.set_offsets(np.c_[x,y])
        self.update_limits()
        self.canvas.draw()
    def update_labels(self,ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.canvas.draw()
        
        
class TrajectoryVisualizer(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer",integration):
        super().__init__()
        self.v = viewer
        self.plot = PlotWindow()
        
        
        
        
        self.xSelection_label = QLabel('X-Axis:')
        self.xSelection = QComboBox()
        self.ySelection_label = QLabel('Y-Axis:')
        self.ySelection = QComboBox()
        self.y2Selection_label = QLabel('Y-Axis 2:')
        self.y2Selection = QComboBox()
        
        self.selectFromLayer = QCheckBox('select from point layer')
        self.trajectorySelection_label = QLabel('Trajectory number:')
        self.trajectorySelection = QSpinBox()
        
        self.init_df(integration)
        self.trajectorySelection.setRange(0,
                                          len(self.df['trajectory'].unique()))
        
        self.xSelection.currentTextChanged.connect(self.replot)
        self.ySelection.currentTextChanged.connect(self.replot)
        self.y2Selection.currentTextChanged.connect(self.replot)
        self.trajectorySelection.valueChanged.connect(self.replot)
        self.trajectorySelection.valueChanged.connect(self.show_active)
        self.selectFromLayer.stateChanged.connect(self.disable_trajectorySelection)
        self.setLayout(QVBoxLayout())
        
        self.layout().addWidget(self.xSelection_label)
        self.layout().addWidget(self.xSelection)
        self.layout().addWidget(self.ySelection_label)
        self.layout().addWidget(self.ySelection)
        self.layout().addWidget(self.y2Selection_label)
        self.layout().addWidget(self.y2Selection)
        self.layout().addWidget(self.selectFromLayer)
        self.layout().addWidget(self.trajectorySelection_label)
        self.layout().addWidget(self.trajectorySelection)
        self.layout().addWidget(self.plot)
        
        self.v.dims.events.current_step.connect(self.plot_current_frame)
        self.v.layers.events.connect(self.plot_highlighted)
        
    def disable_trajectorySelection(self):
        if self.selectFromLayer.isChecked():
            self.trajectorySelection.setEnabled(False)
            self.trajectorySelection_label.setEnabled(False)
        else:
            self.trajectorySelection.setEnabled(True)
            self.trajectorySelection_label.setEnabled(True)
    def plot_highlighted(self, event):
        if event._type == 'highlight':
            if isinstance(self.v.layers.selection.active, Points):
                index = self.v.layers.selection.active._highlight_index
                if isinstance(index, list):
                    index = index[0]
                if self.selectFromLayer.isChecked():
                    self.trajectorySelection.setValue(index)
                    
    def plot_current_frame(self,event):
        current_step = event.value[0]
        
        index = self.trajectorySelection.value()
        traj_selction = self.df['trajectory'].unique()[index]
        traj = self.df[self.df['trajectory']==traj_selction]
        traj_curr = traj[traj[self.xSelection.currentText()]==current_step]
        x = np.array(traj_curr[self.xSelection.currentText()])
        y = np.array(traj_curr[self.ySelection.currentText()])
        self.plot.update_scatter(x, y)
    
    def show_active(self):
        index = self.trajectorySelection.value()
        traj_selction = self.df['trajectory'].unique()[index]
        traj = self.df[self.df['trajectory']==traj_selction]
        x = traj['x'].iloc[0]
        y = traj['y'].iloc[0]
        if 'active trajectory' in self.v.layers:
            points = self.v.layers['active trajectory']
        else:    
            self.v.add_points([x,y],name='active trajectory', face_color='transparent',
                          edge_color='yellow',
                          size=10,
                          edge_width = 0.1)
            points = self.v.layers['active trajectory']
        points.data = [x,y]
    
    def init_df(self,df):   
        self.df = df
        self.xSelection.clear()
        self.xSelection.addItems(self.df.keys())
        self.ySelection.clear()
        self.ySelection.addItems(self.df.keys())
        self.y2Selection.clear()
        self.y2Selection.addItems(self.df.keys())
        self.y2Selection.addItem('None')
    
    def replot(self, s):
        index = self.trajectorySelection.value()
        traj_selction = self.df['trajectory'].unique()[index]
        traj = self.df[self.df['trajectory']==traj_selction]
        current_step = self.v.dims.current_step[0]
        self.plot.update(self.plot.line1,
                         np.array(traj[self.xSelection.currentText()]),
                         np.array(traj[self.ySelection.currentText()]))       
        self.plot.update(self.plot.line2,
                         np.array(traj[self.xSelection.currentText()]),
                         np.array(traj[self.y2Selection.currentText()]))
        traj_curr = traj[traj[self.xSelection.currentText()]==current_step]
        x = np.array(traj_curr[self.xSelection.currentText()])
        y = np.array(traj_curr[self.ySelection.currentText()])
        self.plot.update_scatter(x,y)
        self.plot.update_labels(self.plot.ax,self.xSelection.currentText(),self.ySelection.currentText())
        self.plot.update_labels(self.plot.ax2,self.xSelection.currentText(),self.y2Selection.currentText())
        
            
            
    def show_traj(self,traj,**kwargs):
        xy = np.array(traj[['x','y']].iloc[0])
        self.viewer.add_points(xy,**kwargs)
    def show_all_traj(self,df,**kwargs):
        xy = np.array(df.groupby('trajectory')[['x','y']].first(0))
        self.viewer.add_points(xy,**kwargs)