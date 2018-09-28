import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import random as rnd
import math

import Light
import Shader

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
        self.computeNormals()

    def computeNormals(self):
        for v in range(0, self.nW*self.nH) :
            self.norms[v] = np.array([0.,0.,0.])
        for tri in range(0, len(self.triIdx)) :
            i, j, k = self.triIdx[tri][0], self.triIdx[tri][1], self.triIdx[tri][2]
            vji = self.verts[j] - self.verts[i]
            vki = self.verts[k] - self.verts[i]
            vjiXvki = np.cross(vji, vki)
            self.norms[i] += vjiXvki
            self.norms[j] += vjiXvki
            self.norms[k] += vjiXvki
        for v in range(0, self.nW*self.nH) :
            l = np.linalg.norm(self.norms[v])
            self.norms[v] /= l

    def drawSurface(self):
        glVertexPointer(3, GL_FLOAT, 0, self.verts)
        glEnableClientState(GL_VERTEX_ARRAY)
        glNormalPointer(GL_FLOAT, 0, self.norms)
        glEnableClientState(GL_NORMAL_ARRAY)
        #glDrawArrays(GL_POINTS, 0, self.nParticles)
        glDrawElements(GL_TRIANGLES, (self.nW-1)*(self.nH-1)*2*3, GL_UNSIGNED_INT, self.triIdx)

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

        self.ShaderButton = wx.Button(self, wx.ID_ANY, "Shader On/Off",
                                      pos=(1030, 20), size=(200,40), style = 0)
        self.Bind(wx.EVT_BUTTON, self.OnShaderButton, self.ShaderButton)

    def OnShaderButton(self, event):
        if self.canvas.bDrawWithShader == True :
            self.canvas.bDrawWithShader = False
        else :
            self.canvas.bDrawWithShader = True

class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :
        self.initialized = False
        self.bDrawWithShader = False
        self.shader = None
        self.size = (1024,720)
        self.aspect_ratio = 1
        self.angle = 0.0
        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.mySurface = Surface(30,50)
        self.light = Light.Light()
        self.light.setLight()
        self.light.setMaterial()
        self.light.setLightPoisition()
        self.light.turnOn()
        self.InitGL()

    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 100.0)
        glViewport(0,0,self.size[0], self.size[1])
        self.mySurface.resetVerts()
        glEnable(GL_DEPTH_TEST)
        self.shader = Shader.Shader("vertexSimple", "fragmentSimple")


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

        if self.bDrawWithShader :
            self.shader.begin()

        self.mySurface.drawSurface()

        self.shader.end()

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