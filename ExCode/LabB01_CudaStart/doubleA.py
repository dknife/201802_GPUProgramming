# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
a = numpy.random.randn(50,50)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*8;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(32,8,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print("original array:")
print(a)
print("doubled with kernel:")
print(a_doubled)
