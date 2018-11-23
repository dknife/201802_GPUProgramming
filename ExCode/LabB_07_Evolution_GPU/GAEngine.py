from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np


class GAEngine:
    def __init__(self, nObstacles, nGenes):
        self.nObstacles = nObstacles
        self.nGenes = nGenes

        self.obstacles = (2.0 * (np.random.rand(self.nObstacles * 2, ) - 0.5)).astype(np.float32)
        self.genes = (2.0 * (np.random.rand(self.nGenes * 2 * 3, ) - 0.5)).astype(np.float32)
        self.fitness = (np.zeros(self.nGenes, )).astype(np.float32)
        self.dataForGPU = np.array([self.nObstacles, self.nGenes], dtype=np.int32)

        self.obsGPU = cuda.mem_alloc(self.obstacles.size * self.obstacles.dtype.itemsize)
        self.geneGPU = cuda.mem_alloc(self.genes.size * self.genes.dtype.itemsize)
        self.fitGPU = cuda.mem_alloc(self.fitness.size * self.fitness.dtype.itemsize)
        self.metaData = cuda.mem_alloc(self.dataForGPU.size * self.dataForGPU.dtype.itemsize)

        self.mod = SourceModule("""

                    __device__ float fracf(float x) {
                        return x - floorf(x);
                    }
                    __device__ float random (float s, float t) {
                        return fracf(sinf(s*12.9898 + t*78.233)*43758.5453123);
                    }

                    __device__ float fitness(
                        float p1x, float p1y, float p2x, float p2y, float p3x, float p3y,
                        float *obs, int nObs) {

                        float ox, oy, ux, uy, vx, vy, wx, wy, pux, puy, pvx, pvy, pwx, pwy, cross1, cross2, cross3, fit;

                        ux = p2x-p1x; uy = p2y-p1y;
                        vx = p3x-p1x; vy = p3y-p1y;
                        fit = ux*vy - uy*vx;
                        if(fit<0) fit = -fit;

                        for (int i=0; i<nObs; i++) {
                            ox = obs[i*2+0];
                            oy = obs[i*2+1];
                            ux = p2x-p1x; uy=p2y-p1y;
                            pux = ox-p1x; puy=oy-p1y;
                            vx = p3x-p2x; vy=p3y-p2y;
                            pvx = ox-p2x; pvy=oy-p2y;
                            wx = p1x-p3x; wy=p1y-p3y;
                            pwx = ox-p3x; pwy=oy-p3y;
                            cross1 = ux*puy-uy*pux;
                            cross2 = vx*pvy-vy*pvx;
                            cross3 = wx*pwy-wy*pwx;
                            if( cross1 > 0 && cross2 > 0 && cross3 > 0){ fit /= 2.0; }
                            if( cross1 < 0 && cross2 < 0 && cross3 < 0){ fit /= 2.0; }
                        }

                        if(p1x > 1 || p1x < -1) { fit = 0; }
                        if(p1y > 1 || p1y < -1) { fit = 0; }
                        if(p2x > 1 || p2x < -1) { fit = 0; }
                        if(p2y > 1 || p2y < -1) { fit = 0; }
                        if(p3x > 1 || p3x < -1) { fit = 0; }
                        if(p3y > 1 || p3y < -1) { fit = 0; }

                        return fit;
                    }

                    __global__ void computeFitness(float *obs, float *gene, float *fit, int* metaData) 
                    {
                        const int idx = threadIdx.x + blockDim.x*blockIdx.x;
                        if(idx>metaData[1]) return;

                        fit[idx] = fitness(
                            gene[idx*6], gene[idx*6+1], gene[idx*6+2], gene[idx*6+3], gene[idx*6+4], gene[idx*6+5],
                            obs, metaData[0]);

                    }
                    """)
        return

    def sendDataToDevice(self):
        cuda.memcpy_htod(self.obsGPU, self.obstacles)
        cuda.memcpy_htod(self.geneGPU, self.genes)
        cuda.memcpy_htod(self.fitGPU, self.fitness)
        cuda.memcpy_htod(self.metaData, self.dataForGPU)

    def getDataFromDevice(self):
        cuda.memcpy_dtoh(self.obstacles, self.obsGPU)
        cuda.memcpy_dtoh(self.genes, self.geneGPU)
        cuda.memcpy_dtoh(self.fitness, self.fitGPU)

    def computeFitness(self, gene, obstacles):
        v1 = gene[0:2]
        v2 = gene[2:4]
        v3 = gene[4:6]
        u = v2 - v1
        v = v3 - v1
        fit = np.linalg.norm(np.cross(u, v))

        for i in range(self.nObstacles):
            obstacle = self.obstacles[i * 2:i * 2 + 2]
            u, pu = v2 - v1, obstacle - v1
            v, pv = v3 - v2, obstacle - v2
            w, pw = v1 - v3, obstacle - v3
            cross1 = np.cross(u, pu)
            cross2 = np.cross(v, pv)
            cross3 = np.cross(w, pw)
            if cross1 > 0 and cross2 > 0 and cross3 > 0: fit /= 2.0
            if cross1 < 0 and cross2 < 0 and cross3 < 0: fit /= 2.0

        if v1[0] > 1 or v1[0] < -1: fit = 0
        if v1[1] > 1 or v1[1] < -1: fit = 0
        if v2[0] > 1 or v2[0] < -1: fit = 0
        if v2[1] > 1 or v2[1] < -1: fit = 0
        if v3[0] > 1 or v3[0] < -1: fit = 0
        if v3[1] > 1 or v3[1] < -1: fit = 0

        return fit

    def computeFitnessVector(self):
        for i in range(self.nGenes):
            gene = self.genes[i * 6:i * 6 + 6]
            self.fitness[i] = self.computeFitness(gene, self.obstacles)

    def computeFitnessVectorGPU(self):
        func = self.mod.get_function("computeFitness")
        threadsPerBlock = int(512)
        blocks = int(self.nGenes / threadsPerBlock + 1)
        func(self.obsGPU, self.geneGPU, self.fitGPU, self.metaData, block=(threadsPerBlock, 1, 1), grid=(blocks, 1, 1))

    def rearrangePopulation(self):
        for i in range(int((self.nGenes) / 2)):
            j = self.nGenes - 1 - i
            if self.fitness[i] < self.fitness[j]:
                g1 = np.copy(self.genes[i * 6:i * 6 + 6])
                g2 = np.copy(self.genes[j * 6:j * 6 + 6])
                self.genes[i * 6:i * 6 + 6] = g2
                self.genes[j * 6:j * 6 + 6] = g1

    def crossOver(self):
        childStart = int((self.nGenes) / 2)

        for i in range(0, childStart, 2):
            j = i + 1
            g1 = np.copy(self.genes[i * 6:i * 6 + 6])
            g2 = np.copy(self.genes[j * 6:j * 6 + 6])
            w = np.random.rand(3)
            child = np.copy(g1)
            child[0:2] = (1 - w[0]) * g1[0:2] + w[2] * g2[0:2]
            child[2:4] = (1 - w[1]) * g1[2:4] + w[1] * g2[2:4]
            child[4:6] = (1 - w[2]) * g1[4:6] + w[0] * g2[4:6]
            childIdx = childStart + int(i / 2)
            self.genes[childIdx * 6:childIdx * 6 + 6] = child

        for i in range(int(childStart * 1.5), self.nGenes, 1):
            mut = 2.0 * (np.random.rand(3, ) - 0.5)
            self.genes[i * 6:i * 6 + 2] *= mut[0]
            self.genes[i * 6 + 2:i * 6 + 4] *= mut[1]
            self.genes[i * 6 + 4:i * 6 + 6] *= mut[2]

    def update(self):
        self.sendDataToDevice()
        self.computeFitnessVectorGPU()
        self.getDataFromDevice()

        self.rearrangePopulation()
        self.crossOver()

    def draw(self):
        glColor3f(1, 1, 0)
        glPointSize(4)
        glBegin(GL_POINTS)
        for i in range(self.nObstacles):
            obstacle = self.obstacles[i * 2:i * 2 + 2]
            glVertex2fv(obstacle)
        glEnd()

        glColor3f(0, 0, 1)
        for i in range(self.nGenes):
            glColor3f(self.fitness[i], 0.0, 1 - self.fitness[i])
            glBegin(GL_LINE_LOOP)
            gene = self.genes[i * 6:i * 6 + 6]
            glVertex2fv(gene[0:2])
            glVertex2fv(gene[2:4])
            glVertex2fv(gene[4:6])
            glEnd()


        #glEnableClientState(GL_VERTEX_ARRAY)
        #glVertexPointer(3, GL_FLOAT, 0, self.genes)
        #glDrawArrays(GL_TRIANGLES, 0, 3*self.nGenes)
