from PIL import Image
import numpy as np
from OpenGL.GL import *


class Texture:
    def __init__(self, imageName):
        self.img = Image.open(imageName)
        self.img_data = np.array(list(self.img.getdata()), np.uint8)
        self.W = self.img.size[0]
        self.H = self.img.size[1]

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.W, self.H, 0, GL_RGB, GL_UNSIGNED_BYTE, self.img_data)


    def startTexture(self):
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glEnable(GL_TEXTURE_2D)

    def stopTexture(self):
        glDisable(GL_TEXTURE_2D)

