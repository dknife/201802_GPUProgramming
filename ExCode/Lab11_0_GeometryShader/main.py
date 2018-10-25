from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import Shader
import numpy as np

myShader = None

n = 50
m = 50
t = 0.0
verts = np.zeros(shape=(n*m*3,), dtype=np.float32)

def initVerts() :
    global verts
    idx = 0
    for i in range(0,n) :
        for j in range(0,m) :
            verts[idx*3: idx*3+3] = [float(i)/n - 0.5, 0.0, float(j)/m - 0.5]
            idx += 1


def reshape(w,h) :
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40, w/h, 0.01, 1000.0)
    glViewport(0,0,w,h)

def disp() :
    global n, m, t, myShader, verts
    glClear(GL_COLOR_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(1,1.5,1, 0.,0,0.0, 0,1,0)

    glRotatef(t, 1,1,1);
    t+=0.1

    myShader.begin()
    loc = glGetUniformLocation(myShader.program, "time")
    glUniform1f(loc, t*0.05)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glDrawArrays(GL_POINTS, 0, n*m)

    myShader.end()
    glFlush()


def main() :
    global myShader

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE|GLUT_DEPTH|GLUT_RGB)
    glutInitWindowSize(512,512)
    glutInitWindowPosition(512,0)
    glutCreateWindow(b"Test Window")

    # initialization
    glClearColor(0, 0.0, 0.0, 0)


    # register callbacks
    glutDisplayFunc(disp)
    glutIdleFunc(disp)
    glutReshapeFunc(reshape)
    initVerts()
    myShader = Shader.Shader("particle.vs", "particle.fs", gsFileName="particle.gs3")


    # enter main infinite-loop
    glutMainLoop()


if __name__ == "__main__" :
    main()