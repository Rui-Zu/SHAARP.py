SHAARP.py v1.0.0
================

A validated Python port of the Mathematica ♯SHAARP.si (single-interface reflected SHG) and
♯SHAARP.ml (multilayer / Maker-fringe SHG) packages, merged into one desktop application.

  SHAARP = Second Harmonic Analysis of Anisotropic Rotational Polarimetry

The original Mathematica packages this port reproduces:
  #SHAARP.si (single interface) : https://github.com/Rui-Zu/SHAARP
  #SHAARP.ml (multilayer) : https://github.com/bzw133/SHAARP.ml

RUNNING (no installation needed -- everything is bundled in this folder)
-------
Windows: double-click SHAARP_py.exe
macOS: open SHAARP_py.app. The app is not Apple-signed, so the FIRST launch is blocked:
          - macOS 15 (Sequoia) or newer: try to open it once, then go to System Settings ->
            Privacy & Security and click "Open Anyway".
          - macOS 14 or older: right-click (Control-click) the app -> Open -> Open.
          - Terminal alternative: xattr -dr com.apple.quarantine SHAARP_py.app
Inside the app: Help -> User Guide for the workflow, and hover any control for a tooltip.

YOUR FIRST CALCULATION (about one minute)
-----------------------
1. The app opens on the #SHAARP.si tab. Leave every setting as it is.
2. Under "Case Study and Examples", pick "GaAs (111)" -- one of the four worked cases of the 2022
   paper, listed under "Cases in DOI" (all at 800 nm).
3. Click "Update / Run". The reflected SHG polarimetry I_p(phi) / I_s(phi) polar plots appear next
   to a schematic of the sample.
4. Click the #SHAARP.ml tab, keep the preset "Quartz + Au (Fig 4, 800 nm)", set Functionality to
   "Maker Fringes", and click "Update / Run" -- the transmitted SHG fringes of the 2024 paper's
   Fig-4 heterostructure appear.
5. From there, explore: the case lists carry the complete original case-study palettes at their
   published wavelengths, the papers' heterostructure presets, and an N-layer stack editor for
   your own samples.

WHAT IT COMPUTES
----------------
* Reflected SHG polarimetry I_s(phi), I_p(phi) for any SHG-active crystal class & orientation
* Multilayer Maker fringes under Full multiple reflections / Jerphagnon-Kurtz / Herman-Hayden
* Linear Fresnel reflection/transmission sweeps
* Closed-form analytical SHG expressions (symbolic in polarization, d_ij, thickness)

VALIDATION
----------
Every solver is checked value-by-value against the original live Mathematica SHAARP packages
(agreement typically 1e-9 to 1e-15, case-dependent) and against the published equations in the
references below.

PLEASE CITE
-----------
1. Zu, R., Wang, B., He, J. et al. "Analytical and numerical modeling of optical second harmonic
   generation in anisotropic crystals using #SHAARP package." npj Computational Materials 8, 246
   (2022). doi:10.1038/s41524-022-00930-4
2. Zu, R., Wang, B., He, J. et al. "Optical second harmonic generation in anisotropic multilayers
   with complete multireflection of linear and nonlinear waves using #SHAARP.ml package."
   npj Computational Materials 10, 64 (2024). doi:10.1038/s41524-024-01229-2

Authors: R. Zu, B. Wang, L. Weber, A. Saha, L.-Q. Chen & V. Gopalan (The Pennsylvania State
University). Acknowledgment: U.S. DOE, Office of Science, Basic Energy Sciences, Computational
Materials Sciences Program, Award No. DE-SC0020145.

LICENSE
-------
GNU General Public License v3 (same as the original SHAARP packages) -- see LICENSE.txt.
