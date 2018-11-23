import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

plt.rcParams['toolbar'] = 'None'
fig = plt.figure(figsize=(6, 6), facecolor='white')



def drawComplex(z, color) :
    plt.scatter(z.real, z.imag, c=color)

def drawComplexNumbers(zList, color) :
    for i in range(len(zList)) :
        plt.scatter(zList[i].real, zList[i].imag, c=color)

def drawLineToComplex(z, color) :
    plt.plot([0, z.real], [0, z.imag], color=color, linestyle='-', linewidth=2)

def drawText(z) :
    plt.text(z.real, z.imag, str(z.real) + "+" + str(z.imag)+"$i$")

def drawTextAt(z, string) :
    plt.text(z.real, z.imag, string)


drawLineToComplex(3, 'y')
drawTextAt(3, "REAL")
drawLineToComplex(3j, 'y')
drawTextAt(3j, "IMAGINARY")
drawComplex(0, 'r')



p = 1+3j
drawComplex(p, "b")
drawLineToComplex(1, "b")
drawLineToComplex(p, "b")

for i in range(20, 180, 20) :
    sine = math.sin(math.radians(i))
    cosine = math.cos(math.radians(i))
    R = cosine + sine * 1j
    drawComplex(R*p, "r")
    drawLineToComplex(R, "r")


plt.title("Complex numbers can rotate and scale your coordinates")
plt.grid(True)
plt.show()