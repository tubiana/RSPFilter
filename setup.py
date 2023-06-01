from setuptools import setup, find_packages

MAJOR = 1
MINOR = 1
PATCH = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, PATCH)

with open("RSPFilter/version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(VERSION))


setup(
    name='RSPFilter',
    version=VERSION,
    url='https://github.com/tubiana/RSPFilter',
    license='MIT',
    author='Thibault Tubiana',
    author_email='tubiana.thibault@gmail.com',

    description='RSPFilter: Relion Star Particles file homogeneous filtering.',
    platforms=["Linux", "Solaris", "Mac OS-X", "darwin", "Unix", "win32"],

    install_requires=['numpy',
                      'starfile',
                      'plotly',
                      'scipy',
                      'panel',
                      'pandas',
                      ],

    entry_points={'console_scripts':['RSPFilter=RSPFilter.RSPFilter:main']},


    packages=find_packages(),
)