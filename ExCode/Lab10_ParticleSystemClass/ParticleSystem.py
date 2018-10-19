from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import Shader
import numpy as np

class ParticleSystem :
    def __init__(self, nParticles, shaderFile):
        self.nParticles = nParticles;
        self.cps = Shader.ComputeShader(shaderFile)

        self.f0 = np.random.rand(10000, ).astype('f')
        self.f1 = -self.f0;
        self.out = np.zeros(shape=(10000,), dtype='f')
        self.input0ssbo = glGenBuffers(1)
        self.input1ssbo = glGenBuffers(1)
        self.outputssbo = glGenBuffers(1)

        self.cps.setupShaderStorageBufferObject(self.input0ssbo, 0, self.f0)
        self.cps.setupShaderStorageBufferObject(self.input1ssbo, 1, self.f1)
        self.cps.setupShaderStorageBufferObject(self.outputssbo, 2, self.out)

        self.cps.begin()

        glDispatchCompute(1000, 1, 1)  # arraySize / local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.outputssbo)

        print("particle system ------------------------------")
        print(self.out)

        p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE)
        ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float * len(self.out)))
        array = np.frombuffer(ptr.contents, 'f')

        print("particle system ------------------------------")
        print(array)

        self.cps.end()

    def show(self):

        self.cps.begin()

        self.cps.setupShaderStorageBufferObject(self.input0ssbo, 0, self.f0)
        self.cps.setupShaderStorageBufferObject(self.input1ssbo, 1, self.f1)
        self.cps.setupShaderStorageBufferObject(self.outputssbo, 2, self.out)

        glDispatchCompute(1000, 1, 1)  # arraySize / local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.outputssbo)

        print("particle system ------------------------------")
        print(self.out)

        p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE)
        ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float * len(self.out)))
        array = np.frombuffer(ptr.contents, 'f')
        self.f0 = array

        print("particle system ------------------------------")
        print(array)

        self.cps.end()
