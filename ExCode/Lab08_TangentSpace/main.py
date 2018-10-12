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
import Texture
import Surface

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

        self.shaderButton = wx.Button(self, wx.ID_ANY, "Shader On/Off",
                                      pos=(1030, 20), size=(200,40), style = 0)
        self.shaderLabel = wx.StaticText(self, -1, pos=(1030, 60), style=wx.ALIGN_CENTER)
        self.shaderLabel.SetLabel("currently the shader is off")
        self.Bind(wx.EVT_BUTTON, self.OnShaderButton, self.shaderButton)

        self.lightLabel = wx.StaticText(self, -1, pos=(1030,150), style=wx.ALIGN_CENTER)
        self.lightLabel.SetLabel("Light")
        self.lightSlider = wx.Slider(self, -1, pos=(1030, 180), size = (200,50), style = wx.SL_HORIZONTAL|wx.SL_AUTOTICKS,
                                     value=0, minValue=-20, maxValue=20)

        self.objectRotation = wx.StaticText(self, -1, pos=(1030, 250), style=wx.ALIGN_CENTER)
        self.objectRotation.SetLabel("Object Rotatation (Y)")
        self.objectRotationSlider = wx.Slider(self, -1, pos=(1030, 280), size=(200, 50),
                                     style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                     value=0, minValue=-90, maxValue=90)

        self.Bind(wx.EVT_SLIDER, self.OnLightSlider, self.lightSlider)
        self.Bind(wx.EVT_SLIDER, self.OnRotationSlider, self.objectRotationSlider)

    def OnLightSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.lightX = val / float(10)

    def OnRotationSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.objectAngle = val

    def OnShaderButton(self, event):
        if self.canvas.bDrawWithShader == True :
            self.canvas.bDrawWithShader = False
            self.shaderLabel.SetLabel("currently the shader is off")
        else :
            self.canvas.bDrawWithShader = True
            self.shaderLabel.SetLabel("currently the shader is on")

class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :
        self.initialized = False
        self.bDrawWithShader = False
        self.shader = None
        self.size = (1024,720)
        self.aspect_ratio = 1
        self.lightX = 0.0
        self.objectAngle = 0.0
        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.light = Light.Light()
        self.light.setLight()
        self.light.setMaterial()

        self.light.turnOn()
        self.InitGL()


    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 100.0)
        glViewport(0,0,self.size[0], self.size[1])
        glEnable(GL_DEPTH_TEST)
        self.texture = Texture.Texture("normal.png")
        self.texture.startTexture()
        self.shader = Shader.Shader("tangent.vs", "tangent.fs")
        self.surface = Surface.Surface(30,30)
        self.surface.resetVerts()
        #self.surface.computeNormals()
        self.surface.computeTangentSpace()


    def OnDraw(self, event):
        # clear color and depth buffers
        if not self.initialized :
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position viewers
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,1,1, 0, 0, 0, 0,1,0)
        glRotatef(self.objectAngle, 0,1,0)

        self.light.setLightPoisition(self.lightX, 0.5, 0)

        if self.bDrawWithShader :
            self.shader.begin()
            loc = glGetUniformLocation(self.shader.program, "myTexture")
            glUniform1i(loc, 0)

            vab = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vab)
            glBufferData(GL_ARRAY_BUFFER, 4*self.surface.nParticles, self.surface.tangent, GL_STATIC_DRAW)
            loc = glGetAttribLocation(self.shader.program, "Tangent")
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))


            #loc = glGetAttribLocation(self.shader.program, "Binormal")
            #glEnableVertexAttribArray(self.surface.binor)
            #glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        else : self.surface.drawTangentSpace()

        self.surface.drawSurface()

        self.shader.end()

        glDisable(GL_LIGHTING)
        glPointSize(10)
        glColor3f(1,0,0)
        glBegin(GL_POINTS)
        glVertex3f(self.lightX, 0.5, 0)
        glEnd()
        glEnable(GL_LIGHTING)

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