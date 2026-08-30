# FAQ & troubleshooting

**Why does my chosen material show "SHG ≈ 0 (symmetry-forbidden)"?**
Because its point group is centrosymmetric or isotropic (e.g. Air and Au ∞∞m, Al₂O₃ 6/mmm, Pt m3m,
"Blank linear" m3m). These groups sit in the *— Centrosymmetric (SHG-inactive) —* section of the
**Point group** dropdown; selecting any of them sets $d \equiv 0$ and collapses the *SHG Tensor
dᵢⱼ* group with the note "— not used: SHG-inactive point group (d ≡ 0)". SHG is forbidden by
symmetry for these, so the reflected/film signal is identically zero — the app states this clearly
instead of drawing a spurious curve. There is no separate "SHG active" switch: activity follows
the point group, as in the original package.

**What is the difference between the Full / JK / HH assumptions?**
They are different treatments of multiple reflections in the multilayer Maker sweep: **Full multiple
reflections (FMR)** keeps all multiply-reflected waves (with a forward/backward/standing sub-mode);
**Jerphagnon–Kurtz** drops multiple reflections (single pass); **Herman–Hayden** keeps them only for
the homogeneous $2\omega$ waves. See {doc}`ml_tab`.

**Why do some Maker-fringe curves show a note about interpolated points?**
At isolated incidence angles an eigenmode degeneracy makes a term a removable $0/0$; those points are
masked and interpolated so the curve stays continuous. The title reports how many were interpolated.

**At normal incidence ($\theta_i = 0$) the schematic rays are vertical — is that right?**
Yes. At normal incidence the incident, reflected, and transmitted beams all travel along the surface
normal, so they are drawn vertical (separated slightly for legibility). As $\theta_i$ grows they fan
out to the true entered angle.

**My $d$-extraction is not full rank / some components are unidentifiable.**
A single polarimetry geometry does not always constrain every $d_{ij}$. Combine reflected and
transmitted channels and/or multiple incidence angles or sample azimuths; the extraction result
reports the conditioning/identifiability so you can see which components are well-determined. See
{py:func}`shaarp.extract_si_d_voigt`.

**How noise-robust is $d$-extraction on real (noisy) data?**
It depends strongly on the method. The **field** (phase-resolved) method degrades gracefully —
median error tracks the noise level (≈1.7% recovered error at 2% intensity noise on a dense
multi-geometry scan). The **intensity** (phase-less Gram + rank-1) method amplifies noise
catastrophically at typical conditioning (tens of percent error at 2% noise): treat its output on
noisy data as an initial guess for *which components are large*, not a quantitative estimate. The
seeded Monte-Carlo characterization lives in `benchmarks/dextraction_noise_benchmark.py`.

**The Update seems stuck on an analytical mode.**
It almost certainly is not — the analytical modes run a computer-algebra solve whose *first* run for
a configuration can take minutes (the progress bar says so). The result is cached: repeat Updates on
the same configuration return instantly.

**I entered a wavelength and got an amber note about a "tabulated range".**
Case-study dielectric tensors are interpolated from the original package's exported dispersion
grids. Outside a grid the tensors clamp to the nearest tabulated value; the note names the affected
material and its valid range so you know the tensors are no longer wavelength-accurate there.

**How accurate is SHAARP.py?**
Every solver is validated against the original Mathematica ♯SHAARP package and the published equations,
typically to ~1e-9 — and the published validation figures of both papers are replicated end-to-end
through the GUI compute path (see {doc}`../references`). Full evidence: {doc}`../validation`.

**Can I script this instead of using the GUI?**
Yes — see {doc}`../usage` and the {doc}`../api/index`. The GUI's **Update** simply calls the same
`run_*` facades the API exposes.
