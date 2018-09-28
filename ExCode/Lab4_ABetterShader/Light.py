from OpenGL.GL import *
from OpenGL.GLU import *


class Light :
    def __init__(self):
        self.lDiffuse  = [1.0, 1.0, 1.0, 1.0]  # r, g, b, alpha
        self.lSpecular = [1.0, 1.0, 1.0, 1.0]
        self.lAmbient  = [0.1, 0.1, 0.1, 1.0]
        self.lLocation = [1.0, 1.0, 1.0, 1.0]

        self.mShininess = [64.0]
        self.mDiffuse  = [1.0, 1.0, 0.0, 1.0]
        self.mSpecular = [1.0, 1.0, 1.0, 1.0]
        self.mAmbient  = [0.0, 0.0, 0.0, 1.0]

    def setLight(self):
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.lDiffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, self.lSpecular)
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.lAmbient)
    def setMaterial(self):
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.mDiffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.mSpecular)
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.mAmbient)
        glMaterialfv(GL_FRONT, GL_SHININESS, self.mShininess)

    def setLightPoisition(self):
        glLightfv(GL_LIGHT0, GL_POSITION, self.lLocation)

    def turnOn(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def turnOff(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)


