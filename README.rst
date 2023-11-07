####################################################################
Estimate the magnetic deflection of airshowers in earth's atmosphere
####################################################################

|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |LicenseBadge|

This tool uses KIT's CORSIKA to simulate the airshowers by cosmic-rays
and their emission of Cherenkov-light.
The goal of this tool is to find the:

- direction

and

- position (impact position on the ground w.r.t. to the observer)

a primary particle must have when it enters earth's atmosphere in order for
an observer to see the Cherenkov-light from its extensive air-shower.

At energies above 20GeV the Cherenkov-light of an air-shower is almost parallel
to the direction of the primary particle. In this case, the direction
and position of the primary particle is easy to estimate, as it is just the
trajectory which intersects with the observer.

However, at energies below 10GeV, earth's magnetic field bends the air-shower
so much that the direction of the primary particle significantly differs from
the direction of the air-shower's Cherenkov-emission.


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
