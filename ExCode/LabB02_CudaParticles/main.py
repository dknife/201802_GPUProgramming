import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import random as rnd
import math
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at

import ParticleSystem

class MyFrame(wx.Frame) :
    def __init__(self):
        self.size = (1280, 720)
        wx.Frame.__init__(self, None, title = "wx frame", size = self.size,
                          style = wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.panel = MyPanel(self)

class MyPanel(wx.Panel) :
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.canvas = OpenGLCanvas(self)


        self.resetButton = wx.Button(self, wx.ID_ANY, "Reset",
                                      pos=(1130, 20), size=(100, 40), style=0)
        self.Bind(wx.EVT_BUTTON, self.OnResetButton, self.resetButton)



        self.objectRotation = wx.StaticText(self, -1, pos=(1030, 250), style=wx.ALIGN_CENTER)
        self.objectRotation.SetLabel("Object Rotatation (Y)")
        self.objectRotationSlider = wx.Slider(self, -1, pos=(1030, 280), size=(200, 50),
                                     style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                     value=0, minValue=-90, maxValue=90)

        self.zoomLabel = wx.StaticText(self, -1, pos=(1030, 350), style=wx.ALIGN_CENTER)
        self.zoomLabel.SetLabel("Zoom")
        self.zoomSlider = wx.Slider(self, -1, pos=(1030, 380), size=(200, 50),
                                              style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                              value=10, minValue=1, maxValue=20)

        self.Bind(wx.EVT_SLIDER, self.OnRotationSlider, self.objectRotationSlider)
        self.Bind(wx.EVT_SLIDER, self.OnZoomSlider, self.zoomSlider)

    def OnRotationSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.objectAngle = val

    def OnZoomSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.zoom = val/10.0

    def OnResetButton(self, event):
        self.canvas.particles.initParticles()

class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :
        self.initialized = False
        self.size = (1024,720)
        self.aspect_ratio = 1

        self.objectAngle = 0.0
        self.zoom = 1.0

        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)

        self.particles = ParticleSystem.ParticleSystem(1000, "simulate.cps")

        self.InitGL()


    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 100.0)
        glViewport(0,0,self.size[0], self.size[1])
        glEnable(GL_DEPTH_TEST)


    def OnDraw(self, event):
        # clear color and depth buffers
        if not self.initialized :
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position viewers
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,self.zoom*0.3,self.zoom, 0, 0, 0, 0,1,0)
        glPointSize(10)
        glColor3f(1, 0, 0)
        glBegin(GL_POINTS)
        glEnd()

        glRotatef(self.objectAngle, 0,1,0)

        self.particles.show()

        self.SwapBuffers()

    def OnIdle(self, event):
        self.Refresh()


def main() :
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__" :
    main()