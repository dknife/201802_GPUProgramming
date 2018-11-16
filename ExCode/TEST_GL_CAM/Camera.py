from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math
import numpy as np

def rotate(v, angle, axis) :
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1-c
    x,y,z = axis[0], axis[1], axis[2]

    R1 = np.array([x*x*C+c, x*y*C - z*s, x*z*C+y*s])
    R2 = np.array([y*x*C+z*s, y*y*C+c, y*z*C-x*s])
    R3 = np.array([z*x*C-y*s, z*y*C+x*s, z*z*C+c])

    return np.array([R1.dot(v), R2.dot(v), R3.dot(v)])

class Camera :
    def __init__(self, l, t, u): # 생성자
        self.loc = np.array(l)
        self.tar = np.array(t)

        self.up = np.array(u)
        l = np.linalg.norm(self.up)
        self.up = self.up / l

        self.dir = self.tar - self.loc
        l = np.linalg.norm(self.dir)
        self.dir = self.dir / l

        self.right = np.cross(self.dir, self.up)
        l = np.linalg.norm(self.right)
        self.right = self.right / l

        self.up = np.cross(self.right, self.dir)


        self.asp = 1.0
        self.fov = 60.0
        self.near = 1.0
        self.far = 1000.0




    # various methods
    def setLens(self):
        gluPerspective(self.fov, self.asp, self.near, self.far)

    def setCameraPosition(self):
        gluLookAt(
            self.loc[0], self.loc[1], self.loc[2],
            self.tar[0], self.tar[1], self.tar[2],
            self.up[0], self.up[1], self.up[2]
        )

    def moveForward(self, step = 1.0):
        self.loc = self.loc + self.dir * step
        self.tar = self.tar + self.dir * step

    def moveBackward(self, step = 1.0):
        self.moveForward(-step)


    def moveRight(self, step = 1.0):
        self.loc = self.loc + self.right * step;
        self.tar = self.tar + self.right * step;

    def moveLeft(self, step = 1.0):
        self.moveRight(-step)


    def turnLeft(self, angle=0.1):
        # we must compute new dir, right, tar
        self.dir = rotate(self.dir, angle, self.up)
        self.right = rotate(self.right, angle, self.up)
        self.tar = self.loc + self.dir

    def turnRight(self, angle=0.1):
        self.turnLeft(-angle)

    def turnUp(self, angle=0.1):
        # we must compute new dir, up, tar
        self.dir = rotate(self.dir, angle, self.right);
        self.up = rotate(self.up, angle, self.right)
        self.tar = self.loc + self.dir

    def turnDown(self, angle=0.1):
        self.turnUp(-angle)


    def rollRight(self, angle=0.1):
        # we must compute new up, right
        self.up = rotate(self.up, angle, self.dir)
        self.right = rotate(self.right, angle, self.dir)

    def rollLeft(self, angle=0.1):
        self.rollRight(-angle)