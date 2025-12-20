from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        # Initialize git submodules if they haven't been initialized
        self.initialize_submodules()

        for ext in self.extensions:
            self.build_extension(ext)

    def initialize_submodules(self):
        """Initialize git submodules recursively if not already done."""
        # Check if we're in a git repository
        if not os.path.exists('.git'):
            print("Not in a git repository, skipping submodule initialization")
            return

        # Check if ALL required submodules are initialized (including nested ones)
        required_paths = [
            os.path.join('extlib', 'VieCut', 'lib'),
            os.path.join('extlib', 'VieCut', 'extlib', 'tlx'),  # Nested submodule
            os.path.join('extlib', 'VieCut', 'extlib', 'growt'),  # Another nested submodule
        ]

        all_initialized = all(
            os.path.exists(path) and os.listdir(path)
            for path in required_paths
        )

        if all_initialized:
            print("Submodules already initialized")
            return

        print("Initializing git submodules (including nested submodules)...")
        try:
            subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
            print("Git submodules initialized successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to initialize git submodules: {e}")

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Get pybind11 cmake directory
        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                     '-DPython_EXECUTABLE=' + sys.executable,
                     '-DPYTHON_EXECUTABLE=' + sys.executable,
                     '-Dpybind11_DIR=' + pybind11_cmake_dir,
                     '-DBUILD_PYTHON_MODULE=ON']  # Enable Python module build

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='nekos',
    version='0.1.0',
    author='Vikram Ramavarapu',
    description='Super-lean library for large-scale graph data',
    long_description='Fast C++ graph data structures and I/O utilities with an emphasis on community search and detection',
    packages=find_packages(),
    ext_modules=[CMakeExtension('nekos.nekos', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=['numpy>=1.19.0'],
    extras_require={
        'networkx': ['networkx>=2.0'],
        'networkit': ['networkit>=10.0'],
        'igraph': ['igraph>=0.10.0'],
        'viz': ['vispy>=0.9.0'],
        'ai': ['torch>=1.9.0', 'torchvision>=0.10.0'],
        'all': ['networkx>=2.0', 'networkit>=10.0', 'igraph>=0.10.0', 'vispy>=0.9.0', 'torch>=1.9.0', 'torchvision>=0.10.0'],
    },
)
