####################################################################
Estimate the magnetic deflection of airshowers in earth's atmosphere
####################################################################

|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |LicenseBadge|

This tool uses KIT's CORSIKA to simulate the airshowers by cosmic-rays
and their emission of Cherenkov-light.
The goal of this tool is to find the:

- direction (w.r.t. zenith)

and

- position (w.r.t to observer)

a cosmic-ray must have in order for the observer to see its Cherenkov-light.

At energies > 10GeV the Cherenkov-light of an airshower is almost parallel
to the direction of the primary particle.
However, at energies < 10GeV, earth's magnetic field bends the airshower
so much that an offset occurs.


.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/magnetic_deflection/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/magnetic_deflection/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/magnetic_deflection_cherenkov-plenoscope-project
    :target: https://pypi.org/project/magnetic_deflection_cherenkov-plenoscope-project

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |LicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
