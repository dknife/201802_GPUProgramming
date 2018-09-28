from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math
import shader

myShader = None

n = 300
m = 300
t = 0.0

verts = []

def initVerts(n, m) :
    global verts
    for i in range(0,n) :
        for j in range(0,m) :
            x, z = float(i)/n, float(j)/m
            verts.append(x)
            verts.append(0)
            verts.append(z)


def reshape(w,h) :
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(50, w/h, 0.1, 10000.0)
    glViewport(0,0,w,h)

def disp() :
    global n, m, t, myShader, verts

    if myShader is None :
        myShader = shader.Shader()

    glClear(GL_COLOR_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(1,1,1, 0.5,0,0.5, 0,1,0)

    myShader.begin()

    # we must pass time information to vertex shader here
    loc = glGetUniformLocation(myShader.program, "time")
    glUniform1f(loc, t)

    #glBegin(GL_POINTS)
    #for i in range(0, n*m) :
    #    verts[i*3+1] = 0.2 * math.sin(t*(verts[i*3+0]+verts[i*3+2]));
    #    glVertex3f(verts[i*3],verts[i*3+1],verts[i*3+2])
    #glEnd()
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, verts)
    glDrawArrays(GL_POINTS, 0, (int) (n*m))
    glFinish()
    t += 0.1

    myShader.end()

    glFlush()


def main() :
    global n, m

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE|GLUT_DEPTH|GLUT_RGB)
    glutInitWindowSize(512,512)
    glutInitWindowPosition(512,0)
    glutCreateWindow(b"Test Window")

    # initialization
    glClearColor(0, 0.0, 0.0, 0)


    # init verts
    initVerts(n,m)
    # register callbacks
    glutDisplayFunc(disp)
    glutIdleFunc(disp)
    glutReshapeFunc(reshape)

    # enter main infinite-loop
    glutMainLoop()


if __name__ == "__main__" :
    main()