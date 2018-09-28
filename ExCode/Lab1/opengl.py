from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math
import shader

myShader = None

n = 200
m = 200
t = 0.0

verts = []
idx = []

def initVerts() :
    idxnumber = 0
    for i in range(0,n) :
        for j in range(0,m) :
            x, z = float(i) / n, float(j) / m
            verts.append(x)
            verts.append(0)
            verts.append(z)
            idx.append(idxnumber)
            idxnumber+=1

def reshape(w,h) :
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(50, w/h, 0.1, 10000.0)
    glViewport(0,0,w,h)

def disp() :
    global n, m, t, verts, idx, myShader

    if myShader is None :
        myShader = shader.Shader()

    glClear(GL_COLOR_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(1,1,1, 0.5,0,0.5, 0,1,0)

    myShader.begin()
    loc = glGetUniformLocation(myShader.program, "time")
    glUniform1f(loc, t)

    glVertexPointer(3, GL_FLOAT, 0, verts)
    glEnableClientState(GL_VERTEX_ARRAY)
    glDrawElements(GL_POINTS, n*m, GL_UNSIGNED_INT, idx);

    t += 0.1

    myShader.end()

    glFlush()


def main() :
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE|GLUT_DEPTH|GLUT_RGB)
    glutInitWindowSize(512,512)
    glutInitWindowPosition(512,0)
    glutCreateWindow(b"Test Window")

    # initialization
    glClearColor(0, 0.0, 0.0, 0)
    initVerts()

    # register callbacks
    glutDisplayFunc(disp)
    glutIdleFunc(disp)
    glutReshapeFunc(reshape)

    # enter main infinite-loop
    glutMainLoop()


if __name__ == "__main__" :
    main()