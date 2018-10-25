from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import Shader
import numpy as np

class ParticleSystem :
    def __init__(self, nParticles, shaderFile):
        self.nParticles = nParticles;
        self.cps = Shader.ComputeShader(shaderFile)

        self.colors = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)
        self.locIn = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)
        self.velIn = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)
        self.locOut = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)
        self.velOut = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)

        self.initParticles()

        print("particle system: buffer set ")
        self.locInSSO = glGenBuffers(1)
        self.velInSSO = glGenBuffers(1)
        self.locOutSSO = glGenBuffers(1)
        self.velOutSSO = glGenBuffers(1)

    def initParticles(self) :
        self.locIn = np.zeros(shape=(self.nParticles * 3,), dtype=np.float32)
        self.velIn = np.zeros(shape=(self.nParticles * 3,), dtype=np.float32)
        for i in range(self.nParticles) :
            self.locIn[i*3:i*3+3] = [0, np.random.rand(1)*0.25+0.25, 0] #[0.1 * (np.random.rand(1) - 0.5), 0.5, 0.1 * (np.random.rand(1) - 0.5)]
            self.velIn[i*3:i*3+3] = [(np.random.rand(1) - 0.5), -0.5, (np.random.rand(1) - 0.5)]
            self.colors[i*3:i*3+3] = [0.5*(np.random.rand(1)+1.0), 0.5*np.random.rand(1), 0.5*(np.random.rand(1)+1.0)]

    def show(self):

        self.cps.setupShaderStorageBufferObject(self.locInSSO, 0, self.locIn)
        self.cps.setupShaderStorageBufferObject(self.velInSSO, 1, self.velIn)
        self.cps.setupShaderStorageBufferObject(self.locOutSSO, 2, self.locOut)
        self.cps.setupShaderStorageBufferObject(self.velOutSSO, 3, self.velOut)

        self.cps.begin()

        loc = glGetUniformLocation(self.cps.program, "nParticles")
        glUniform1i(loc, self.nParticles)
        glDispatchCompute( int(self.nParticles*3), 1,1)  # arraySize / local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.locOutSSO)
        p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE)
        ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float * self.nParticles*3))
        self.locOut = np.frombuffer(ptr.contents, 'f')

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.velOutSSO)
        p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE)
        ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float * self.nParticles*3))
        self.velOut = np.frombuffer(ptr.contents, 'f')

        self.cps.end()

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glColor3f(0,1,0)
        glPointSize(5)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.locOut)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        glDrawArrays(GL_POINTS, 0, self.nParticles)
        self.velIn = self.velOut
        self.locIn = self.locOut
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
