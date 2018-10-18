from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math

class Surface :
    def __init__(self, nW, nH):
        self.nParticles = nW*nH
        self.nW, self.nH = nW, nH
        self.verts = np.zeros(shape=(self.nParticles, 3), dtype = np.float32)
        self.norms = np.zeros(shape=(self.nParticles, 3), dtype = np.float32)
        #########################
        self.tangent= np.zeros(shape=(self.nParticles, 3), dtype = np.float32)
        self.binorm = np.zeros(shape=(self.nParticles, 3), dtype = np.float32)
        #########################
        self.tex = np.zeros(shape=(self.nParticles, 2), dtype=np.float32)
        self.triIdx = np.zeros(shape=((self.nW-1)*(self.nH-1)*2, 3), dtype=np.int32)


    def resetVerts(self):
        for i in range(0, self.nW) :
            for j in range(0, self.nH) :
                x = self.verts[j * self.nW + i, 0] = float(i) / (self.nW-1) - 0.5
                z = self.verts[j * self.nW + i, 2] = float(j) / (self.nH - 1) - 0.5
                #self.verts[j * self.nW + i, 1] = rnd.randint(0,100)/5000.0 + 0.05*math.sin(20.5*(x*z+z+x))
                self.verts[j * self.nW + i, 1] = 0.08 * math.sin(8.5 * (x * z + z + x))
                self.tex[j * self.nW + i, 0] = float(i) / (self.nW - 1)
                self.tex[j * self.nW + i, 1] = 1.0 - float(j) / (self.nH - 1)


        triangleNumber = 0
        for col in range(0, self.nW-1) :
            for row in range(0, self.nH-1) :
                idx = row*self.nW + col
                self.triIdx[triangleNumber] = np.array([idx, idx+self.nW, idx+1])
                triangleNumber += 1
                self.triIdx[triangleNumber] = np.array([idx + 1, idx + self.nW, idx + self.nW + 1])
                triangleNumber += 1
        self.computeNormals()

    def computeNormals(self):
        for v in range(0, self.nW*self.nH) :
            self.norms[v] = np.array([0.,0.,0.])
        for tri in range(0, len(self.triIdx)) :
            i, j, k = self.triIdx[tri][0], self.triIdx[tri][1], self.triIdx[tri][2]
            vji = self.verts[j] - self.verts[i]
            vki = self.verts[k] - self.verts[i]
            vjiXvki = np.cross(vji, vki)
            self.norms[i] += vjiXvki
            self.norms[j] += vjiXvki
            self.norms[k] += vjiXvki
        for v in range(0, self.nW*self.nH) :
            l = np.linalg.norm(self.norms[v])
            self.norms[v] /= l

    def computeTangentSpace(self):
        for v in range(0, self.nW*self.nH) :
            self.norms[v] = np.array([0.,0.,0.])
            self.binorm[v] = np.array([0.,0.,0.])
            self.tangent[v] = np.array([0.,0.,0.])

        for tri in range(0, len(self.triIdx)) :
            p1, p2, p3 = self.triIdx[tri][0], self.triIdx[tri][1], self.triIdx[tri][2]
            u1, v1 = self.tex[p1][0], self.tex[p1][1]
            u2, v2 = self.tex[p2][0], self.tex[p2][1]
            u3, v3 = self.tex[p3][0], self.tex[p3][1]
            x21 = self.verts[p2] - self.verts[p1]
            x31 = self.verts[p3] - self.verts[p1]
            v31, v21 = v3-v1, v2-v1
            u31, u21 = u3-u1, u2-u1
            T =  ( v31*x21 - v21*x31 ) / (u21*v31-v21*u31)
            B =  ( u31*x21 - u21*x31 ) / (v21*u31-u21*v31)
            #T = T / np.linalg.norm(T)
            #B = B / np.linalg.norm(B)

            self.tangent[p1] += T
            self.tangent[p2] += T
            self.tangent[p3] += T
            self.binorm[p1] += B
            self.binorm[p2] += B
            self.binorm[p3] += B
            N = np.cross(T,B)
            self.norms[p1] += N
            self.norms[p2] += N
            self.norms[p3] += N

        for v in range(0, self.nW*self.nH) :
            l = np.linalg.norm(self.norms[v])
            self.norms[v] /= l
            l = np.linalg.norm(self.tangent[v])
            self.tangent[v] /= l
            l = np.linalg.norm(self.binorm[v])
            self.binorm[v] /= l



    def drawSurface(self):
        glVertexPointer(3, GL_FLOAT, 0, self.verts)
        glEnableClientState(GL_VERTEX_ARRAY)
        glNormalPointer(GL_FLOAT, 0, self.norms)
        glEnableClientState(GL_NORMAL_ARRAY)
        glTexCoordPointer(2, GL_FLOAT, 0, self.tex)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glDrawElements(GL_TRIANGLES, (self.nW-1)*(self.nH-1)*2*3, GL_UNSIGNED_INT, self.triIdx)

    def drawTangentSpace(self):
        glLineWidth(3)
        glDisable(GL_LIGHTING)
        for v in range(0, self.nW*self.nH) :
            glBegin(GL_LINES)
            glColor3f(1, 0, 0)
            glVertex3fv(self.verts[v])
            glVertex3fv(self.verts[v] + 0.05 * self.tangent[v])
            glColor3f(0, 1, 0)
            glVertex3fv(self.verts[v])
            glVertex3fv(self.verts[v] + 0.05 * self.binorm[v])
            glColor3f(0,0,1)
            glVertex3fv(self.verts[v])
            glVertex3fv(self.verts[v]+0.05*self.norms[v])
            glEnd()
        glEnable(GL_LIGHTING)