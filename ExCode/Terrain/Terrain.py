from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import numpy as np

class Terrain :

    def __init__(self):
        self.nV = 0
        self.mx =  100000000.0
        self.Mx = -100000000.0
        self.my = 100000000.0
        self.My = -100000000.0
        self.mz = 100000000.0
        self.Mz = -100000000.0

        self.nW = 0
        self.nH = 0
        self.dx = 0
        self.dz = 0



    def load(self, filename):

        with open(filename, "rt") as terrain :
            self.nV = int(next(terrain))
            self.dx = float(next(terrain))
            self.dz = float(next(terrain))
            self.verts = np.zeros(shape=(self.nV*3,), dtype='f')
            self.color = np.zeros(shape=(self.nV*3,), dtype='f')

            print(self.nV)
            for i in range(self.nV) :
                x,z,y = next(terrain).split()
                x = float(x)
                y = float(y) * 6.0
                z = float(z)
                self.verts[i*3:i*3+3] = [x,y,z]
                g = y/120.0
                if g > 1.0 : g = 1.0
                self.color[i*3:i*3+3] = [0.5*(1-g), g+0.5, 1-g]
                if x > self.Mx: self.Mx = x
                if x < self.mx: self.mx = x
                if y > self.My: self.My = y
                if y < self.my: self.my = y
                if z > self.Mz: self.Mz = z
                if z < self.mz: self.mz = z

        xLen, zLen = self.Mx - self.mx, self.Mz - self.mz
        self.nW = int (xLen / self.dx) + 1
        self.nH = int (zLen / self.dz) + 1

        self.nMeshParticles = self.nW*self.nH

        self.meshV = np.zeros(shape=(self.nMeshParticles, 3), dtype=np.float32)
        self.meshN = np.zeros(shape=(self.nMeshParticles, 3), dtype=np.float32)
        self.meshC = np.zeros(shape=(self.nMeshParticles, 3), dtype=np.float32)
        self.meshTri = np.zeros(shape=((self.nW - 1) * (self.nH - 1) * 2, 3), dtype=np.int32)

        for i in range(0, self.nW):
            for j in range(0, self.nH):
                self.meshV[j * self.nW + i, 0] = self.mx + self.dx * i
                self.meshV[j * self.nW + i, 2] = self.mz + self.dz * j
                self.meshC[j * self.nW + i] = [0,0,0]

        for i in range(self.nV):
            v = self.verts[i*3:i*3+3]
            idxX = int ((v[0]-self.mx)/self.dx)
            idxZ = int ((v[2] - self.mz) / self.dz)
            print(idxX, idxZ)
            self.meshV[idxZ * self.nW + idxX, 1] = v[1]
            self.meshC[idxZ * self.nW + idxX] = self.color[i*3:i*3+3]

        triangleNumber = 0
        for col in range(0, self.nW - 1):
            for row in range(0, self.nH - 1):
                idx = row * self.nW + col
                self.meshTri[triangleNumber] = np.array([idx, idx + self.nW, idx + 1])
                triangleNumber += 1
                self.meshTri[triangleNumber] = np.array([idx + 1, idx + self.nW, idx + self.nW + 1])
                triangleNumber += 1

        self.computeMeshNormals()

    def computeMeshNormals(self):
        for v in range(0, self.nW * self.nH):
            self.meshN[v] = np.array([0., 0., 0.])
        for tri in range(0, len(self.meshTri)):
            i, j, k = self.meshTri[tri][0], self.meshTri[tri][1], self.meshTri[tri][2]
            vji = self.meshV[j] - self.meshV[i]
            vki = self.meshV[k] - self.meshV[i]
            vjiXvki = np.cross(vji, vki)
            self.meshN[i] += vjiXvki
            self.meshN[j] += vjiXvki
            self.meshN[k] += vjiXvki
        for v in range(0, self.nW * self.nH):
            l = np.linalg.norm(self.meshN[v])
            self.meshN[v] /= l

    def v(self, idx):
        return self.verts[idx * 3:idx * 3 + 3]


    def drawPoints(self):

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.verts)
        glColorPointer(3, GL_FLOAT, 0, self.color)
        glDrawArrays(GL_POINTS, 0, self.nV)

    def drawMesh(self):
        glVertexPointer(3, GL_FLOAT, 0, self.meshV)
        glEnableClientState(GL_VERTEX_ARRAY)
        glNormalPointer(GL_FLOAT, 0, self.meshN)
        glEnableClientState(GL_NORMAL_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, self.meshC)
        glEnableClientState(GL_COLOR_ARRAY)
        glDrawElements(GL_TRIANGLES, (self.nW - 1) * (self.nH - 1) * 2 * 3, GL_UNSIGNED_INT, self.meshTri)
