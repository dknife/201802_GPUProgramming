import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import math

import MatrixShow
matrixShow = MatrixShow.MatrixShow()

class MyFrame(wx.Frame) :
    def __init__(self):
        self.size = (1280, 720)
        wx.Frame.__init__(self, None, title = "Young-Min Kang's Lecuture Labs for Matrix", size = self.size,
                          style = wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.panel = MyPanel(self)

class MyPanel(wx.Panel) :
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.canvas = OpenGLCanvas(self)

        self.view2DButton = wx.Button(self, wx.ID_ANY, "View 2D",
                                      pos=(1030, 20), size=(100,40), style = 0)
        self.Bind(wx.EVT_BUTTON, self.OnView2DButton, self.view2DButton)

        self.camHeight = wx.StaticText(self, -1, pos=(1030, 150), style=wx.ALIGN_CENTER)
        self.camHeight.SetLabel("Camera Height")
        self.camHeightSlider = wx.Slider(self, -1, pos=(1030, 180), size=(200, 50),
                                     style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                     value=0, minValue=-50, maxValue=50)

        self.objectRotation = wx.StaticText(self, -1, pos=(1030, 250), style=wx.ALIGN_CENTER)
        self.objectRotation.SetLabel("Object Rotatation (Y)")
        self.objectRotationSlider = wx.Slider(self, -1, pos=(1030, 280), size=(200, 50),
                                     style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                     value=0, minValue=-90, maxValue=90)

        self.zoomLabel = wx.StaticText(self, -1, pos=(1030, 350), style=wx.ALIGN_CENTER)
        self.zoomLabel.SetLabel("Zoom")
        self.zoomSlider = wx.Slider(self, -1, pos=(1030, 380), size=(200, 50),
                                              style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                              value=10, minValue=1, maxValue=200)

        self.Bind(wx.EVT_SLIDER, self.OnCamHeightSlider, self.camHeightSlider)
        self.Bind(wx.EVT_SLIDER, self.OnRotationSlider, self.objectRotationSlider)
        self.Bind(wx.EVT_SLIDER, self.OnZoomSlider, self.zoomSlider)

    def OnView2DButton(self, event):
        self.canvas.objectAngle = 0.0
        self.canvas.camHeight = 0.0

    def OnCamHeightSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.camHeight = val/10.0

    def OnRotationSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.objectAngle = val

    def OnZoomSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.zoom = val/10.0


class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :
        self.initialized = False
        self.size = (1024,720)
        self.aspect_ratio = 1

        self.objectAngle = 0.0
        self.zoom = 1.0
        self.camHeight = 1.0

        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)

        self.InitGL()


    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 1000.0)
        glViewport(0,0,self.size[0], self.size[1])

        glClearColor(0.25, 0.25, 0.25, 1.0)


    def OnDraw(self, event):
        global matrixShow
        # clear color and depth buffers
        if not self.initialized :
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position viewers
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,self.camHeight*self.zoom,self.zoom, 0, 0, 0, 0,1,0)

        glRotatef(self.objectAngle, 0,1,0)

        self.drawAxis()

        matrixShow.OnDraw()

        self.SwapBuffers()

    def OnIdle(self, event):
        matrixShow.OnIdle()
        self.Refresh()

    def drawAxis(self):
        glLineWidth(2)
        glColor3f(0.25, 0.15, 0.25)
        glBegin(GL_LINES)
        for i in range(-10,11,1):
            glVertex3f(i, 0, -10)
            glVertex3f(i, 0,  10)
        for i in range(-10, 11, 1):
            glVertex3f(-10, 0, i)
            glVertex3f( 10, 0, i)
        glColor3f(0.2, 0.2, 0.25)
        for i in range(-10, 11, 1):
            glVertex3f(i, -10, 0)
            glVertex3f(i, 10, 0)
        for i in range(-10, 11, 1):
            glVertex3f(-10, i, 0)
            glVertex3f(10, i, 0)
        glColor3f(1,0,0)
        glVertex3f(0, 0, 0)
        glVertex3f(10, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 10, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 10)
        glEnd()




def main() :
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__" :
    main()