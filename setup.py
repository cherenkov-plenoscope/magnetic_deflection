import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='magnetic_deflection',
    version='0.0.1',
    description='Explore magnetic deflection of cosmic-rays below 10GeV.',
    long_description=long_description,
    url='https://github.com/cherenkov-plenoscope',
    author='Sebastian Achim Mueller',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    license='GPL v3',
    packages=[
        'magnetic_deflection',
    ],
    install_requires=[
        'corsika_primary_wrapper',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
