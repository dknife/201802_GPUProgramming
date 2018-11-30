from OpenGL.GL import *
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

        self.rSeeds = np.array([1.0, 1.0], dtype=np.float32)

        self.obsGPU = cuda.mem_alloc(self.obstacles.size * self.obstacles.dtype.itemsize)
        self.geneGPU = cuda.mem_alloc(self.genes.size * self.genes.dtype.itemsize)
        self.fitGPU = cuda.mem_alloc(self.fitness.size * self.fitness.dtype.itemsize)
        self.metaData = cuda.mem_alloc(self.dataForGPU.size * self.dataForGPU.dtype.itemsize)

        self.rSeedsGPU = cuda.mem_alloc(self.rSeeds.size * self.rSeeds.dtype.itemsize)

        self.mod = SourceModule(open("cudaKernels.cu", "r").read())
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

    def setRandomeSeed(self):
        self.rSeeds = np.array([np.random.rand(1)+1.0,np.random.rand(1)+1.0], dtype=np.float32)
        cuda.memcpy_htod(self.rSeedsGPU, self.rSeeds)

    def computeFitnessVectorGPU(self):
        func = self.mod.get_function("computeFitness")
        threadsPerBlock = int(512)
        blocks = int(self.nGenes / threadsPerBlock + 1)
        func(self.obsGPU, self.geneGPU, self.fitGPU, self.metaData, block=(threadsPerBlock, 1, 1), grid=(blocks, 1, 1))

    def shuffleGPU(self):
        nThreads = int(self.nGenes / 2)
        func = self.mod.get_function("shuffleGene")
        threadsPerBlock = int(512)
        blocks = int(nThreads / threadsPerBlock + 1)
        func(self.geneGPU, self.fitGPU, self.rSeedsGPU,  self.metaData, block=(threadsPerBlock, 1, 1), grid=(blocks, 1, 1))

    def rearrangePopulationGPU(self):
        nThreads = int(self.nGenes / 2)
        func = self.mod.get_function("rearrangePopulation")
        threadsPerBlock = int(512)
        blocks = int(nThreads / threadsPerBlock + 1)
        func(self.geneGPU, self.fitGPU, self.metaData, block=(threadsPerBlock, 1, 1), grid=(blocks, 1, 1))

    def crossOverGPU(self):
        self.setRandomeSeed()

        nThreads = int(self.nGenes / 4)
        func = self.mod.get_function("crossOver")
        threadsPerBlock = int(512)
        blocks = int(nThreads / threadsPerBlock + 1)
        func(self.geneGPU, self.rSeedsGPU, self.metaData, block=(threadsPerBlock, 1, 1), grid=(blocks, 1, 1))

    def mutateGPU(self):
        nThreads = int(self.nGenes / 4)
        func = self.mod.get_function("mutate")
        threadsPerBlock = int(512)
        blocks = int(nThreads / threadsPerBlock + 1)
        func(self.geneGPU, self.rSeedsGPU,  self.metaData, block=(threadsPerBlock, 1, 1), grid=(blocks, 1, 1))



    def update(self):
        self.sendDataToDevice()

        self.rearrangePopulationGPU()
        self.crossOverGPU()
        self.mutateGPU()
        self.shuffleGPU()
        self.computeFitnessVectorGPU()
        self.getDataFromDevice()

    def draw(self):
        glColor3f(0, 0, 1)
        glLineWidth(1)
        bestFit = 0
        bestFitIdx = 0
        for i in range(self.nGenes):
            if self.fitness[i] > bestFit:
                bestFit = self.fitness[i]
                bestFitIdx = i
                print(i, bestFit)

            glColor3f(self.fitness[i], 0.0, 1 - self.fitness[i])
            glBegin(GL_LINE_LOOP)
            gene = self.genes[i * 6:i * 6 + 6]
            glVertex2fv(gene[0:2])
            glVertex2fv(gene[2:4])
            glVertex2fv(gene[4:6])
            glEnd()

        glLineWidth(5)
        glColor3f(0,1,0)
        glBegin(GL_LINE_LOOP)
        gene = self.genes[bestFitIdx * 6:bestFitIdx * 6 + 6]
        glVertex2fv(gene[0:2])
        glVertex2fv(gene[2:4])
        glVertex2fv(gene[4:6])
        glEnd()

        #glEnableClientState(GL_VERTEX_ARRAY)
        #glVertexPointer(3, GL_FLOAT, 0, self.genes)
        #glDrawArrays(GL_TRIANGLES, 0, 3 * self.nGenes)

        glColor3f(1, 1, 0)
        glPointSize(6)
        glBegin(GL_POINTS)
        for i in range(self.nObstacles):
            obstacle = self.obstacles[i * 2:i * 2 + 2]
            glVertex2fv(obstacle)
        glEnd()
