SHAARP.py v1.0.0
================

SHAARP.py simulates and fits optical second-harmonic generation (SHG) in anisotropic crystals
and thin-film multilayers. This folder is the standalone desktop app: nothing else to install,
and no Mathematica or other commercial license needed.

  SHAARP = Second Harmonic Analysis of Anisotropic Rotational Polarimetry

The app has two tabs, one per SHAARP method:
  SHAARP.si  reflected SHG polarimetry from a single crystal surface
  SHAARP.ml  SHG from a multilayer stack: Maker fringes, Fresnel curves, polarimetry

Documentation: https://shaarp-py.readthedocs.io
Source code:   https://github.com/Rui-Zu/SHAARP.py

RUNNING
-------
Windows: extract this zip first (right-click the .zip, then "Extract All..."), open the extracted
         SHAARP_py folder and double-click SHAARP_py.exe. The app needs the _internal folder
         beside it, so it will not start from inside the zip.
macOS:   extract the zip, then open SHAARP_py.app. This build is for Apple Silicon (M-series)
         Macs; on an Intel Mac, run SHAARP.py from Python instead (see the documentation).

The app is not code-signed, so Windows and macOS ask you to allow it the first time you open it,
and the first launch takes a little while with nothing on screen. Later launches are quick. If it
does not open, the FAQ in the documentation has the fix for each system.

Inside the app: the Help menu has the User Guide, and every control shows a tooltip when you
hover over it.

YOUR FIRST CALCULATION (a few minutes)
--------------------------------------
1. The app opens on the SHAARP.si tab. Leave every setting as it is.
2. Under "Case Study and Examples", pick "GaAs (111)", one of the four worked cases of the 2022
   paper (all at 800 nm).
3. Click "Update / Run" at the foot of the input panel (the "Update" button in the toolbar does
   the same thing). The reflected SHG polarimetry I_p(phi) / I_s(phi) polar plots appear next
   to a schematic of the sample.
4. Click the SHAARP.ml tab, keep the preset "Quartz + Au (Fig 4, 800 nm)", set Functionality to
   "Maker Fringes", and click "Update / Run". The transmitted SHG fringes of the 2024 paper's
   Fig-4 heterostructure appear. The sweep is fine enough to resolve them, so give it about
   a minute; raise the theta step for a quicker look at the envelope.
5. From there, explore: the case lists carry the published case studies at their published
   wavelengths, the papers' heterostructure presets, an N-layer stack editor for your own
   samples, and a My Materials palette for materials you define yourself.

WHAT IT COMPUTES
----------------
* Reflected SHG polarimetry I_s(phi), I_p(phi) for any SHG-active crystal class and orientation
* Multilayer Maker fringes under Full multiple reflections / Jerphagnon-Kurtz / Herman-Hayden
* Linear Fresnel reflection/transmission sweeps
* Wavelength sweeps: an SHG spectrum I(lambda), or Maker fringes and Fresnel curves mapped
  against both wavelength and incidence angle, for five crystals with published index data
* Closed-form analytical SHG expressions (symbolic in polarization, d_ij, thickness)
* d-tensor extraction: recover d_ij from a simulated or measured polarimetry scan

The same solvers are available as a Python package (pip install from the repository) for
scripting, batch runs and fitting your own data.

HOW IT IS TESTED
----------------
Every solver is checked against the published equations and the reference output of the
original SHAARP packages, and the test suite runs on every change. What is covered, and how, is
on the documentation site under "How SHAARP.py is tested".

ORIGINS AND CITATION
--------------------
SHAARP.py grew out of two Mathematica packages from the same group:
  SHAARP.si (single interface) : https://github.com/Rui-Zu/SHAARP
  SHAARP.ml (multilayer)       : https://github.com/bzw133/SHAARP.ml

If you use SHAARP.py, please cite the papers that introduced the methods:
1. Zu, R., Wang, B., He, J. et al. "Analytical and numerical modeling of optical second harmonic
   generation in anisotropic crystals using #SHAARP package." npj Computational Materials 8, 246
   (2022). https://doi.org/10.1038/s41524-022-00930-4
2. Zu, R., Wang, B., He, J. et al. "Optical second harmonic generation in anisotropic multilayers
   with complete multireflection of linear and nonlinear waves using #SHAARP.ml package."
   npj Computational Materials 10, 64 (2024). https://doi.org/10.1038/s41524-024-01229-2

Authors: R. Zu, B. Wang, L. Weber, A. Saha, L.-Q. Chen & V. Gopalan (The Pennsylvania State
University). Acknowledgment: U.S. DOE, Office of Science, Basic Energy Sciences, Computational
Materials Sciences Program, Award No. DE-SC0020145.

LICENSE
-------
GNU General Public License v3 -- see LICENSE.txt.
