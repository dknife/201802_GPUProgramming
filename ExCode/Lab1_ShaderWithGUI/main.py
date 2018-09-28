import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import random as rnd
import math

class Surface :
    def __init__(self, nW, nH):
        self.nParticles = nW*nH
        self.nW, self.nH = nW, nH
        self.verts = np.zeros(shape=(self.nParticles, 3), dtype = np.float32)
        self.norms = np.zeros(shape=(self.nParticles, 3), dtype = np.float32)
        self.triIdx = np.zeros(shape=((self.nW-1)*(self.nH-1)*2, 3), dtype=np.int32)

    def resetVerts(self):
        for i in range(0, self.nW) :
            for j in range(0, self.nH) :
                x = self.verts[j * self.nW + i, 0] = float(i) / (self.nW-1) - 0.5
                z = self.verts[j * self.nW + i, 2] = float(j) / (self.nH - 1) - 0.5
                self.verts[j * self.nW + i, 1] = rnd.randint(0,100)/5000.0 + 0.1*math.sin(7.5*(x*z+z+x))


        triangleNumber = 0
        for col in range(0, self.nW-1) :
            for row in range(0, self.nH-1) :
                idx = row*self.nW + col
                self.triIdx[triangleNumber] = np.array([idx, idx+self.nW, idx+1])
                triangleNumber += 1
                self.triIdx[triangleNumber] = np.array([idx + 1, idx + self.nW, idx + self.nW + 1])
                triangleNumber += 1

    def drawSurface(self):
        glVertexPointer(3, GL_FLOAT, 0, self.verts)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawArrays(GL_POINTS, 0, self.nParticles)
        #glDrawElements(GL_TRIANGLES, (self.nW-1)*(self.nH-1)*2*3, GL_UNSIGNED_INT, self.triIdx)



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

        self.x_slider = wx.Slider(self, -1, pos=(1130, 180), size=(40,150), style = wx.SL_VERTICAL|wx.SL_AUTOTICKS,
                                  value=0, minValue=-5, maxValue = 5)

class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :
        self.initialized = False
        self.size = (1024,720)
        self.aspect_ratio = 1
        self.angle = 0.0
        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.mySurface = Surface(30,50)
        self.InitGL()

    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 100.0)
        glViewport(0,0,self.size[0], self.size[1])
        self.mySurface.resetVerts()

    def OnDraw(self, event):
        # clear color and depth buffers
        if not self.initialized :
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position viewers
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1,1,1, 0, 0, 0, 0,1,0)
        glRotatef(self.angle, 0,1,0)

        self.mySurface.drawSurface()

        self.SwapBuffers()

    def OnIdle(self, event):
        self.angle += 1.0
        self.Refresh()


def main() :
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__" :
    main()