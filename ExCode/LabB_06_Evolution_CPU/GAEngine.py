from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np

class GAEngine :
    def __init__(self, nObstacles, nGenes):
        self.nObstacles = nObstacles
        self.nGenes = nGenes

        self.obstacles = (2.0 * (np.random.rand(self.nObstacles * 2,) - 0.5)).astype(np.float32)
        self.genes = (2.0*(np.random.rand(self.nGenes * 2 * 3,)-0.5)).astype(np.float32)
        self.fitness = (np.zeros(self.nGenes, )).astype(np.float32)
        self.dataForGPU = np.array([self.nObstacles, self.nGenes], dtype=np.int32)

        return

    def computeFitness(self, gene, obstacles):
        v1 = gene[0:2]
        v2 = gene[2:4]
        v3 = gene[4:6]
        u = v2-v1
        v = v3-v1
        fit = np.linalg.norm(np.cross(u,v))

        for i in range(self.nObstacles) :
            obstacle = self.obstacles[i * 2:i * 2 + 2]
            u, pu = v2-v1, obstacle-v1
            v, pv = v3-v2, obstacle-v2
            w, pw = v1-v3, obstacle-v3
            cross1 = np.cross(u, pu)
            cross2 = np.cross(v, pv)
            cross3 = np.cross(w, pw)
            if cross1 > 0 and cross2 > 0 and cross3 > 0: fit /= 2.0
            if cross1 < 0 and cross2 < 0 and cross3 < 0: fit /= 2.0

        if v1[0] > 1 or v1[0] < -1 : fit = 0
        if v1[1] > 1 or v1[1] < -1: fit = 0
        if v2[0] > 1 or v2[0] < -1 : fit = 0
        if v2[1] > 1 or v2[1] < -1: fit = 0
        if v3[0] > 1 or v3[0] < -1 : fit = 0
        if v3[1] > 1 or v3[1] < -1: fit = 0

        return fit


    def computeFitnessVector(self):
        for i in range(self.nGenes) :
            gene = self.genes[i*6:i*6+6]
            self.fitness[i] = self.computeFitness(gene, self.obstacles)

    def rearrangePopulation(self):
        for i in range(int((self.nGenes)/2)) :
            j = self.nGenes-1-i
            if self.fitness[i] < self.fitness[j] :
                g1 = np.copy(self.genes[i * 6:i * 6 + 6])
                g2 = np.copy(self.genes[j * 6:j * 6 + 6])
                self.genes[i * 6:i * 6 + 6] = g2
                self.genes[j * 6:j * 6 + 6] = g1

    def crossOver(self):
        childStart = int((self.nGenes)/2)

        for i in range(0,childStart,2) :
            j = i+1
            g1 = np.copy(self.genes[i * 6:i * 6 + 6])
            g2 = np.copy(self.genes[j * 6:j * 6 + 6])
            w = np.random.rand(3)
            child = np.copy(g1)
            child[0:2] = (1 - w[0]) * g1[0:2] + w[2] * g2[0:2]
            child[2:4] = (1 - w[1]) * g1[2:4] + w[1] * g2[2:4]
            child[4:6] = (1 - w[2]) * g1[4:6] + w[0] * g2[4:6]
            childIdx = childStart + int(i/2)
            self.genes[ childIdx * 6:childIdx * 6 + 6] = child

        for i in range(int(childStart*1.5), self.nGenes, 1) :
            mut = 2.0*(np.random.rand(3,)-0.5)
            self.genes[i * 6    :i * 6 + 2] *= mut[0]
            self.genes[i * 6 + 2:i * 6 + 4] *= mut[1]
            self.genes[i * 6 + 4:i * 6 + 6] *= mut[2]



    def update(self):
        self.computeFitnessVector()
        self.rearrangePopulation()
        self.crossOver()

    def draw(self):
        glColor3f(1,1,0)
        glPointSize(4)
        glBegin(GL_POINTS)
        for i in range(self.nObstacles) :
            obstacle = self.obstacles[i*2:i*2+2]
            glVertex2fv(obstacle)
        glEnd()

        glColor3f(0,0,1)
        for i in range(self.nGenes) :
            glColor3f(self.fitness[i], 0.0, 1-self.fitness[i])
            glBegin(GL_LINE_LOOP)
            gene = self.genes[i*6:i*6+6]
            glVertex2fv(gene[0:2])
            glVertex2fv(gene[2:4])
            glVertex2fv(gene[4:6])
            glEnd()