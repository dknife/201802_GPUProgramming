from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import random as rnd
import math

# coding: utf-8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


##############################################################################
# Shader
##############################################################################
# Checks for GL posted errors after appropriate calls
def printOpenGLError():
    err = glGetError()
    if (err != GL_NO_ERROR):
        print('GLERROR: ', gluErrorString(err))
        # sys.exit()


class Shader:
    def __init__(self, vsFileName, fsFileName, gsFileName = None, attrib_list = None):
        self.initShader(vsFileName, fsFileName, gsFileName, attrib_list)

    def initShader(self, vs_file, fs_file, gs_file, attrib_list):
        fileVS = open(vs_file, "r")
        fileFS = open(fs_file, "r")



        # create program
        self.program = glCreateProgram()
        print('create program')
        printOpenGLError()

        # vertex shader
        print('compile vertex shader...')
        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, fileVS.read())
        glCompileShader(self.vs)
        glAttachShader(self.program, self.vs)
        printOpenGLError()

        if gs_file is not None:
            fileGS = open(gs_file, "r")
            print('compile geometry shader...')
            self.gs = glCreateShader(GL_GEOMETRY_SHADER)
            glShaderSource(self.gs, fileGS.read())
            glCompileShader(self.gs)
            glAttachShader(self.program, self.gs)
            printOpenGLError()

        # fragment shader
        print('compile fragment shader...')
        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, fileFS.read())
        glCompileShader(self.fs)
        glAttachShader(self.program, self.fs)
        printOpenGLError()

        if attrib_list is not None :
            for i in range(len(attrib_list)) :
                glBindAttribLocation(self.program, 10+i, attrib_list[i])

        print('link...')
        glLinkProgram(self.program)
        printOpenGLError()

    def begin(self):
        if glUseProgram(self.program):
            printOpenGLError()

    def end(self):
        glUseProgram(0)

class ComputeShader:
    def __init__(self, cpsFileName):
        self.initShader(cpsFileName)

    def initShader(self, cpsFileName):
        fileCPS = open(cpsFileName, "r")

        # create program
        self.program = glCreateProgram()
        print('create program for compute shader')
        printOpenGLError()

        # compute shader load, compile and attach
        print('compile compute shader...')
        self.cps = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(self.cps, fileCPS.read())
        glCompileShader(self.cps)
        glAttachShader(self.program, self.cps)
        printOpenGLError()

        print('link compute shader...')
        glLinkProgram(self.program)
        printOpenGLError()

    def setupShaderStorageBufferObject(self, ssbo, index, bufferData):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, bufferData, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo)

    def begin(self):
        if glUseProgram(self.program):
            printOpenGLError()

    def end(self):
        glUseProgram(0)