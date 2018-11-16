from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math

import Camera


angle = 0.0

myCam = Camera.Camera([10,10,10], [0,0,0], [0,1,0])

def drawPlanet(distance, angle, planetRadius, spin=0.0, slope=0.0) :

    glLineWidth(3)
    glRotatef(slope, 0, 0, 1)
    glBegin(GL_LINE_STRIP)
    for i in range(0, 360):
        theta = 2.0 * 3.141592 * i / 360.0
        x = distance * math.cos(theta)
        y = distance * math.sin(theta)
        glVertex3f(x, 0, y)
    glEnd()



    glRotatef(angle, 0, 1, 0)
    glTranslatef(distance, 0, 0)
    glPushMatrix()
    glRotatef(spin, 0,1,0)
    glutWireSphere(planetRadius, 8, 8)
    glPopMatrix()
    glLineWidth(1)


def key(k, x, y) :
    if k == b'w' :
        myCam.moveForward()
    if k == b's' :
        myCam.moveBackward()
    if k == b'd' :
        myCam.moveRight()
    if k == b'a' :
        myCam.moveLeft()
    if k == b'j' :
        myCam.turnLeft()
    if k == b'l' :
        myCam.turnRight()
    if k == b'i' :
        myCam.turnUp()
    if k == b'k' :
        myCam.turnDown()
    if k == b'.' :
        myCam.rollRight()
    if k == b',' :
        myCam.rollLeft()

def disp() :
    global angle

    # reset buffer
    glClear(GL_COLOR_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    myCam.setLens()

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPushMatrix()
    glTranslatef(0, 0, -1)
    glutWireCube(0.5)
    glPopMatrix()

    myCam.setCameraPosition()



    # drawing

    # sun
    glColor3f(1,0,0)
    glutWireSphere(1.0, 20, 20)

    angle += 0.5

    #내행성
    # venus
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.0)
    drawPlanet(5.0, 2.13*angle, 0.25, slope = 10 )
    glPopMatrix()

    # earth
    glPushMatrix()
    glColor3f(0,0.5,1.0)
    drawPlanet(10.0, angle, 0.5, angle*33.33, slope = -5)

    # moon
    glColor3f(1.0, 1.0, 1.0)
    drawPlanet(2.0, 14.231*angle, 0.1, slope = 90)
    glPopMatrix()

    # 외행성

    # jupiter
    glPushMatrix()
    glColor3f(0.8,0.8,0.5)
    drawPlanet(20.0, 0.2312*angle, 0.5, slope = -30)
    # 목달 1
    glPushMatrix()
    glColor3f(1, 1, 1)
    drawPlanet(4.0, 7.0*angle, 0.2, slope = 50)
    glPopMatrix()
    # 목달 2
    glColor3f(1,1,1)
    drawPlanet(5.0, angle, 0.2, slope = -90)
    glColor3f(1, 0, 0)
    drawPlanet(2.5, 12.23*angle, 0.2, slope = 50)
    glPopMatrix()
    glFlush()


# windowing
glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB)
glutInitWindowSize(512,512)
glutInitWindowPosition(512,0)
glutCreateWindow(b"Test Window")

glClearColor(0, 0.0, 0.0, 0)


# register callbacks
glutDisplayFunc(disp)
glutIdleFunc(disp)
glutKeyboardFunc(key)

# enter main infinite-loop
glutMainLoop()