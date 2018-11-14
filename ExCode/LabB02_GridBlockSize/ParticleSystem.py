from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import Shader

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np


class ParticleSystem :
    def __init__(self, nParticles, shaderFile):

        self.nParticles = nParticles;

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
                const int idx = threadIdx.x + blockDim.x*blockIdx.x;
                if(idx>maxP[0]) return;
                float gravity[] = {0.0, -9.8, 0.0};
                float dt = 0.01;
                
                for(int i=0;i<3;i++) {
                    vel[idx*3+i] = vel[idx*3+i] + gravity[i]*dt;
                    loc[idx*3+i] = loc[idx*3+i]+ vel[idx*3+i]*dt;
                }    
                if (loc[idx*3+1] < 0) {
                    loc[idx*3+1] = -0.7*loc[idx*3+1];
                    vel[idx*3+1] = -0.7*vel[idx*3+1];
                }
                float d = loc[idx*3]*loc[idx*3]+loc[idx*3+2]*loc[idx*3+2];
                if(d>0.5) {
                    vel[idx*3] = random(loc[idx*3], loc[idx*3+2]) - 0.5;
                    vel[idx*3+1] = random(loc[idx*3]*loc[idx*3+2], loc[idx*3+2]);
                    vel[idx*3+2] = random(loc[idx*3+2], loc[idx*3]) - 0.5;
                    loc[idx*3] = 0.0;
                    loc[idx*3+1] = 0.5;
                    loc[idx*3+2] = 0.0;
                }
            }
            """)

        self.initParticles()



    def initParticles(self) :
        self.loc = np.zeros(shape=(self.nParticles * 3,), dtype=np.float32)
        self.vel = np.zeros(shape=(self.nParticles * 3,), dtype=np.float32)
        for i in range(self.nParticles) :
            self.loc[i*3:i*3+3] = [0, np.random.rand(1)*0.25+0.25, 0] #[0.1 * (np.random.rand(1) - 0.5), 0.5, 0.1 * (np.random.rand(1) - 0.5)]
            self.vel[i*3:i*3+3] = [(np.random.rand(1) - 0.5), -0.5, (np.random.rand(1) - 0.5)]
            self.colors[i*3:i*3+3] = [np.random.rand(1), np.random.rand(1), np.random.rand(1)]

    def update(self):
        cuda.memcpy_htod(self.loc_gpu, self.loc)
        cuda.memcpy_htod(self.vel_gpu, self.vel)
        cuda.memcpy_htod(self.maxP_gpu, self.maxP)

        func = self.mod.get_function("updateParticle")

        func(self.loc_gpu, self.vel_gpu, self.maxP_gpu, block=(1024,1,1), grid=(1024,1,1))

        cuda.memcpy_dtoh(self.loc, self.loc_gpu)
        cuda.memcpy_dtoh(self.vel, self.vel_gpu)



    def show(self):


        self.update()

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glColor3f(0,1,0)
        glPointSize(1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.loc)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        glDrawArrays(GL_POINTS, 0, self.nParticles)
        glEnable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
