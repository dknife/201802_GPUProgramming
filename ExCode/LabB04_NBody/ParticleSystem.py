from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import Shader
import Texture


class ParticleSystem :
    def __init__(self, nParticles):

        self.nParticles = nParticles;

        self.particleShader = Shader.Shader("particle.vs", "particle.fs", gsFileName="particle.gs")
        self.sprite = Texture.Texture("sprite.png", option=GL_RGB)

        self.colors = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)  # single precision
        self.loc = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)
        self.vel = np.zeros(shape=(self.nParticles*3,), dtype=np.float32)
        self.maxP = np.array([nParticles], dtype=np.int32)


        self.loc_gpu = cuda.mem_alloc(self.loc.size * self.loc.dtype.itemsize)
        self.vel_gpu = cuda.mem_alloc(self.vel.size * self.vel.dtype.itemsize)
        self.maxP_gpu = cuda.mem_alloc(self.maxP.size * self.maxP.dtype.itemsize)

        self.mod = SourceModule("""
        
            __device__ float fracf(float x) {
                return x - floorf(x);
            }
            
            __device__ float random (float s, float t) {
                return fracf(sinf(s*12.9898 + t*78.233)*43758.5453123);
            }

            __global__ void updateParticle(float *loc, float *vel, int* maxP) 
            { 
                const int idx  = threadIdx.x + blockDim.x*blockIdx.x ;
                if(idx > maxP[0]*maxP[0]) return;
                
                const int idx0 = idx / maxP[0];
                const int idx1 = idx % maxP[0];
                
                if(idx0>maxP[0]) return;
                if(idx1>maxP[0]) return;
                float dt = 0.003;
                
                for(int i=0;i<3;i++) {
                    loc[idx0*3+i] = loc[idx0*3+i]+ vel[idx0*3+i]*dt;
                }
            }
            """)

        self.initParticles()



    def initParticles(self) :
        #self.loc = np.zeros(shape=(self.nParticles * 3,), dtype=np.float32)
        self.loc = np.random.rand(self.nParticles * 3,)
        self.loc = self.loc.astype(dtype=np.float32)
        self.vel = np.random.rand(self.nParticles * 3, )
        self.vel = self.loc.astype(dtype=np.float32)
        for i in range(self.nParticles) :
            self.loc[i*3:i*3+3] -= np.array([0.5, 0.5, 0.5])
            self.vel[i*3:i*3+3] -= np.array([0.5, 0.5, 0.5])
            self.colors[i*3:i*3+3] = [1.0, 0.5, 0.5] #np.random.rand(1), np.random.rand(1), np.random.rand(1)]

    def sendDataToDevice(self):
        cuda.memcpy_htod(self.loc_gpu, self.loc)
        cuda.memcpy_htod(self.vel_gpu, self.vel)
        cuda.memcpy_htod(self.maxP_gpu, self.maxP)

    def getDataFromDevice(self):
        cuda.memcpy_dtoh(self.loc, self.loc_gpu)
        cuda.memcpy_dtoh(self.vel, self.vel_gpu)

    def update(self):
        func = self.mod.get_function("updateParticle")
        func(self.loc_gpu, self.vel_gpu, self.maxP_gpu, block=(1024,1,1), grid=(1000,1,1))


    def show(self):

        self.sprite.startTexture()

        #self.particleShader.begin()
        #loc = glGetUniformLocation(self.particleShader.program, "sprite")
        #glUniform1i(loc, 0)
        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
        glDepthMask(GL_FALSE)
        #glDisable(GL_LIGHTING)
        #glDisable(GL_TEXTURE_2D)

        glPointSize(1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.loc)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        glDrawArrays(GL_POINTS, 0, self.nParticles)

        #glEnable(GL_LIGHTING)
        #glEnable(GL_TEXTURE_2D)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        self.particleShader.end()

        self.sprite.stopTexture()
