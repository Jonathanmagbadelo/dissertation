from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("ngrams", ["n_grams.pyx"])
]
setup(
    ext_modules=cythonize(extensions)
)
