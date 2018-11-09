import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import random as rnd
import math
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at

import Light
import Shader
import Texture
import Surface

import ParticleSystem

class MyFrame(wx.Frame) :
    def __init__(self):
        self.size = (1280, 720)
        wx.Frame.__init__(self, None, title = "Compute-Vertex-Geometry-Fragment Full Demonstration - Young-Min Kang", size = self.size,
                          style = wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.panel = MyPanel(self)

class MyPanel(wx.Panel) :
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.canvas = OpenGLCanvas(self)

        self.shaderButton = wx.Button(self, wx.ID_ANY, "Shader On/Off",
                                      pos=(1030, 20), size=(100,40), style = 0)
        self.shaderLabel = wx.StaticText(self, -1, pos=(1030, 60), style=wx.ALIGN_CENTER)
        self.shaderLabel.SetLabel("currently the shader is off")
        self.Bind(wx.EVT_BUTTON, self.OnShaderButton, self.shaderButton)

        self.resetButton = wx.Button(self, wx.ID_ANY, "Reset",
                                      pos=(1130, 20), size=(100, 40), style=0)
        self.Bind(wx.EVT_BUTTON, self.OnResetButton, self.resetButton)


        self.lightLabel = wx.StaticText(self, -1, pos=(1030,150), style=wx.ALIGN_CENTER)
        self.lightLabel.SetLabel("Light")
        self.lightSlider = wx.Slider(self, -1, pos=(1030, 180), size = (200,50), style = wx.SL_HORIZONTAL|wx.SL_AUTOTICKS,
                                     value=0, minValue=-20, maxValue=20)

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

        self.Bind(wx.EVT_SLIDER, self.OnLightSlider, self.lightSlider)
        self.Bind(wx.EVT_SLIDER, self.OnRotationSlider, self.objectRotationSlider)
        self.Bind(wx.EVT_SLIDER, self.OnZoomSlider, self.zoomSlider)

    def OnLightSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.lightX = val / float(10)

    def OnRotationSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.objectAngle = val

    def OnZoomSlider(self, event):
        val = event.GetEventObject().GetValue()
        self.canvas.zoom = val/10.0

    def OnShaderButton(self, event):
        if self.canvas.bDrawWithShader == True :
            self.canvas.bDrawWithShader = False
            self.shaderLabel.SetLabel("currently the shader is off")
        else :
            self.canvas.bDrawWithShader = True
            self.shaderLabel.SetLabel("currently the shader is on")

    def OnResetButton(self, event):
        self.canvas.particles.initParticles()

class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :
        self.initialized = False
        self.bDrawWithShader = False
        self.shader = None
        self.size = (1024,720)
        self.aspect_ratio = 1

        self.lightX = 0.0
        self.objectAngle = 0.0
        self.zoom = 1.0

        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.light = Light.Light()
        self.light.setLight()
        self.light.setMaterial()


        self.light.turnOn()

        self.texture = Texture.Texture("normal.png")
        attrib_list = ["Tangent", "Binormal"]
        self.shader = Shader.Shader("textureMapping.vs", "textureMapping.fs", attrib_list=attrib_list)




        self.surface = Surface.Surface(50,50)
        self.surface.resetVerts()
        self.surface.computeTangentSpace()

        self.particles = ParticleSystem.ParticleSystem(40000, "simulate.cps")

        self.InitGL()


    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 100.0)
        glViewport(0,0,self.size[0], self.size[1])
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 0.0)



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
        self.light.setLightPoisition(self.lightX, 0.5, 0)
        glDisable(GL_LIGHTING)
        glPointSize(10)
        glColor3f(1, 0, 0)
        glBegin(GL_POINTS)
        glVertex3f(self.lightX, 0.5, 0)
        glEnd()
        glEnable(GL_LIGHTING)

        glRotatef(self.objectAngle, 0,1,0)

        self.texture.startTexture()
        if self.bDrawWithShader :
            self.shader.begin()
            loc = glGetUniformLocation(self.shader.program, "myTexture")
            glUniform1i(loc, 0)
            glVertexAttribPointer(10, 3, GL_FLOAT, GL_FALSE, 0, self.surface.tangent)
            glEnableVertexAttribArray(10)
            glVertexAttribPointer(11, 3, GL_FLOAT, GL_FALSE, 0, self.surface.binorm)
            glEnableVertexAttribArray(11)


        self.surface.drawSurface()
        self.texture.stopTexture()

        self.shader.end()

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