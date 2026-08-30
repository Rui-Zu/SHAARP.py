"""Re-export of the Fig-4 quartz + Au case study, which now lives in the package itself.

Kept so existing imports (`from benchmarks.quartz_au_docs_case import ...`) continue to work. The
definitions moved to :mod:`shaarp.quartz_au_docs_case` so that the preset is available to anyone
who installed SHAARP.py with pip, without a checkout of this repository.
"""
from shaarp.quartz_au_docs_case import *  # noqa: F401,F403
from shaarp.quartz_au_docs_case import (  # noqa: F401
    _gold_material,
    _quartz_material,
    build_quartz_au_case,
    build_quartz_au_system,
    quartz_sellmeier_index,
    AU_N_2OMEGA,
    AU_N_OMEGA,
    AU_THICKNESS_UM,
    D11_PM_V,
    D14_PM_V,
    QUARTZ_THICKNESS_UM,
    WAVELENGTH_UM,
    _QUARTZ_EXTRAORDINARY,
    _QUARTZ_ORDINARY,
)
