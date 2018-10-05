import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *

from PIL import Image
import numpy as np
import math
import Shader

def loadImage(imageName) :
    img = Image.open(imageName)
    img_data = np.array(list(img.getdata()), np.uint8)
    return img.size[0], img.size[1], img_data




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
        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)




    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        glOrtho(-self.aspect_ratio, self.aspect_ratio, -1, 1, -1, 1)
        glViewport(0,0,self.size[0], self.size[1])
        glEnable(GL_DEPTH_TEST)
        self.shader = Shader.Shader("vertex.vs", "smooth.fs")
        self.imgW, self.imgH, myImage = loadImage("baboon.png")
        print("image loaded (w,h) = ", self.imgW, self.imgH)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.imgW, self.imgH, 0, GL_RGB, GL_UNSIGNED_BYTE, myImage)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glEnable(GL_TEXTURE_2D)



    def OnDraw(self, event):
        # clear color and depth buffers
        if not self.initialized :
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position viewers
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.bDrawWithShader :
            self.shader.begin()
            loc = glGetUniformLocation(self.shader.program, "myTexture")
            glUniform1i(loc, 0);
            loc = glGetUniformLocation(self.shader.program, "imgW")
            glUniform1i(loc, self.imgW);
            loc = glGetUniformLocation(self.shader.program, "imgH")
            glUniform1i(loc, self.imgH);


        glBegin(GL_QUADS)
        glTexCoord2fv([0, 1])
        glVertex2fv([-1, -1])
        glTexCoord2fv([1, 1])
        glVertex2fv([1, -1])
        glTexCoord2fv([1, 0])
        glVertex2fv([1, 1])
        glTexCoord2fv([0, 0])
        glVertex2fv([-1, 1])
        glEnd()

        self.shader.end()

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