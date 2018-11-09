from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from math import *

import numpy as np

import Terrain

terrain = Terrain.Terrain()
terrain.load("data.txt")

l_position = [1.0, 1.0, 1.0, 0.0]
time = 0.0


def setLightPosition():
    global l_position

    glLightfv(GL_LIGHT0, GL_POSITION, l_position)


def setLightAndMaterial():
    l_ambient = [0.0, 0.0, 0.0]
    l_diffuse = [0.5, 0.5, 0.5]
    l_specular = [0.5, 0.5, 0.5]

    m_ambient = [1.0, 1.0, 1.0]
    m_diffuse = [1.0, 1.0, 1.0]
    m_specular = [0.3, 0.3, 0.3]
    m_shine = [127.0]

    # Light Setting
    glLightfv(GL_LIGHT0, GL_AMBIENT, l_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, l_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, l_specular)

    glMaterialfv(GL_FRONT, GL_AMBIENT, m_ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, m_diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, m_specular)
    glMaterialfv(GL_FRONT, GL_SHININESS, m_shine)


def myReshape(w, h):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspRatio = w / h
    gluPerspective(60, aspRatio, 1, 1000000)
    glViewport(0, 0, w, h)


def myDisplay():
    global time, terrain

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    x = (terrain.Mx + terrain.mx ) / 2.0
    y = (terrain.My + terrain.my) / 2.0
    z = (terrain.Mz + terrain.mz) / 2.0
    dx = terrain.Mx - terrain.mx
    gluLookAt(dx*0.5,terrain.My*3,dx*0.5,0,0,0,0,1,0)

    time += 0.05
    l_position[0] = sin(time*0.1)


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glRotatef(time*10, 0, 1, 0)
    glTranslatef(-x, -y, -z)
    setLightPosition()
    terrain.drawMesh()

    glFlush()

    return


def GLInit():
    # clear color setting
    glClearColor(0, 0.15, 0.15, 1)
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    glEnable(GL_COLOR_MATERIAL)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    setLightAndMaterial()


def main(arg):
    # opengl glut initialization
    glutInit(arg)

    # window setting
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(600, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Let There Be Light")

    GLInit()

    glutReshapeFunc(myReshape)
    glutDisplayFunc(myDisplay)
    glutIdleFunc(myDisplay)

    glutMainLoop()


if __name__ == "__main__":
    main(sys.argv)