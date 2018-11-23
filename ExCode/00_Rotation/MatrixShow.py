import wx # requires wxPython package
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

class MatrixShow:
    def __init__(self):
        self.time = 0
        self.R =np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        self.I = np.identity(3)

    def OnIdle(self):
        self.time = self.time + 0.1
        c = math.cos(self.time)
        s = math.sin(self.time)

        self.R = np.array([
            [c,-s,c],
            [s,c,s],
            [s,s,1]
        ])


    def OnDraw(self):
        o = np.array([0,0])
        points = np.array([
            [1,1, 0.5],
            [0.5,0.5, 0.5],
            [0.1,0.3,0.7],
            [0.2, 0.1, 0.2],
            [0.4, 0.5, 0.3],
            [0.2, 0.7, 0.7]
        ])
        v = np.array([1,1,1])
        for i in range(len(points)):
            self.drawPoint(points[i], [0,1,0])
            pTemp = self.R.dot(points[i])
            self.drawPoint(pTemp, [0, 1, 0])


        self.drawMatrix(self.R)
        self.drawMatrix(self.I)
        return

    def drawPoint(self, p, color=[0,0,0]):
        glPointSize(10)
        glColor3fv(color)
        glBegin(GL_POINTS)
        if len(p) == 2 :
            glVertex2fv(p)
        else :
            glVertex3fv(p)
        glEnd()

    def drawLine(self, p0, p1, color=[0,0,0]):
        glLineWidth(2)
        glColor3fv(color)
        glBegin(GL_LINES)
        if len(p0) == 2 :
            glVertex2fv(p0)
        else :
            glVertex3fv(p0)
        if len(p0) == 2 :
            glVertex2fv(p1)
        else :
            glVertex3fv(p1)
        glEnd()


    def drawVector(self, v, color=[0,0,0]):
        glLineWidth(2)
        glColor3fv(color)
        glBegin(GL_LINES)
        glVertex3f(0,0,0);
        if len(v) == 2 :
            glVertex2fv(v)
        else :
            glVertex3fv(v)
        glEnd()

    def drawMatrix(self, M, color=[0.5,0.5,0.5]):
        if len(M[0]) == 2 :
            u = np.array([M[0, 0], M[1, 0]])
            v = np.array([M[0, 1], M[1, 1]])
            self.drawLine([0, 0, 0], u, [0.5, 0,0])
            self.drawLine([0, 0, 0], v, [0,0.5,0])
            self.drawLine(u, u + v, color)
            self.drawLine(v, u + v, color)
        else :
            u = np.array([M[0, 0], M[1, 0], M[2,0]])
            v = np.array([M[0, 1], M[1, 1], M[2,1]])
            w = np.array([M[0, 2], M[1, 2], M[2,2]])
            self.drawLine([0, 0, 0], u, [0.5, 0,0])
            self.drawLine([0, 0, 0], v, [0,0.5,0])
            self.drawLine([0, 0, 0], w, [0,0,0.5])
            self.drawLine(u, u+v, color)
            self.drawLine(v, u+v, color)
            self.drawLine(w, u+w, color)
            self.drawLine(w, v+w, color)
            self.drawLine(u+w, u + v+w, color)
            self.drawLine(v+w, u + v+w, color)
            self.drawLine(u, u + w, color)
            self.drawLine(v, v + w, color)
            self.drawLine(u+v, u + v+w, color)

