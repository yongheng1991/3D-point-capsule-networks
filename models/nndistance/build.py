from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CppExtension,CUDAExtension


#setup(name='my_lib',
#      ext_modules=[CppExtension('my_lib', ['src/my_lib.cpp'])],
#      cmdclass={'build_ext': BuildExtension})

setup(name='my_lib_cuda',
      ext_modules=[CUDAExtension('my_lib_cuda',['src/my_lib_cuda.cpp', 'src/nnd_cuda.cu']
              )],
      cmdclass={'build_ext': BuildExtension}
      )


#if __name__ == '__main__':
#    ffi.build()
