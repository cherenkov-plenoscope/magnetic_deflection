import os
import setuptools

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("magnetic_deflection", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="magnetic_deflection_cherenkov-plenoscope-project",
    version=version,
    description="Explore magnetic deflection of cosmic-rays below 10GeV.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/magnetic_deflection",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "magnetic_deflection",
        "magnetic_deflection.dome",
        "magnetic_deflection.allsky",
    ],
    package_data={
        "magnetic_deflection": [os.path.join("scripts", "*.py")],
    },
    install_requires=[
        "corsika_primary",
        "rename_after_writing",
        "atmospheric_cherenkov_response_cherenkov-plenoscope-project",
        "json_utils_sebastian-achim-mueller",
        "json_line_logger>=0.0.2",
        "binning_utils_sebastian-achim-mueller>=0.0.11",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
