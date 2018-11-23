import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *

import GAEngine

class MyFrame(wx.Frame) :
    def __init__(self):
        self.size = (1280, 720)
        wx.Frame.__init__(self, None, title = "Genetic Algorithm", size = self.size,
                          style = wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.panel = MyPanel(self)

class MyPanel(wx.Panel) :
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.canvas = OpenGLCanvas(self)


        self.runGAButton = wx.Button(self, wx.ID_ANY, "Runn GA",
                                      pos=(1030, 20), size=(100,40), style = 0)
        self.Bind(wx.EVT_BUTTON, self.OnrunGAButton, self.runGAButton)


    def OnrunGAButton(self, event):
        self.canvas.bStart = True


class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent) :

        self.initialized = False
        self.size = (1024,720)
        self.aspect_ratio = 1
        self.bStart = False

        glcanvas.GLCanvas.__init__(self, parent, -1, size = self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)
        self.Bind(wx.EVT_IDLE, self.OnIdle)

        nObstacles = 20
        nGenes = 1000

        self.gaEngine = GAEngine.GAEngine(nObstacles, nGenes)

        self.InitGL()


    def InitGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        glOrtho(-1,1,-1,1,-1,1)
        glViewport(0,0,self.size[0], self.size[1])
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


    def OnDraw(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)

        # clear color and depth buffers
        if not self.initialized :
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # position viewers
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.gaEngine.draw()

        self.SwapBuffers()

    def OnIdle(self, event):

        if self.bStart == True:
            self.gaEngine.update()

        self.Refresh()


def main() :
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__" :
    main()