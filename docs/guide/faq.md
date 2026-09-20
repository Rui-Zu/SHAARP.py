# FAQ & troubleshooting

## Getting the app to open

**The app does not open at all on Windows — no window, no error.**
The zip was probably not extracted. Windows lets you double-click an `.exe` from inside a zip, but
it unpacks only that one file, and the app needs the `_internal` folder beside it. Right-click the
`.zip`, choose **Extract All…**, then run `SHAARP_py\SHAARP_py.exe` from the extracted folder. If a
window still does not appear, give the first launch 10–30 seconds: the system scans the bundle
before it starts, and later launches are quick.

**Windows says "Windows protected your PC".**
The app is not code-signed, so this box appears once. Click **More info**, then **Run anyway**.

**macOS says the app cannot be opened because it is from an unidentified developer.**
Same reason: the app is not Apple-signed, and macOS blocks the first open only. On macOS 15 and
newer, try to open the app once, then open **System Settings**, go to **Privacy & Security**, and
click **Open Anyway**. On macOS 14 and older, Control-click the app, choose **Open**, and confirm
with **Open**. From a terminal, `xattr -dr com.apple.quarantine SHAARP_py.app` does the same. The
first launch also takes 10–30 seconds with nothing on screen.

**macOS says the app "is not supported on this Mac".**
That is an Intel Mac. The packaged macOS build is Apple Silicon (M-series) only, and it is not a
universal binary. Run SHAARP.py from Python instead — see {doc}`install_launch`, Option 2.

## Physics and results

**Why does the Quartz + Au Maker-fringe curve look jagged rather than like smooth fringes?**
Because the angle step is too coarse for it. That heterostructure's SHG-active layer is a 121.2 µm
z-cut quartz plate, whose Maker fringes sit about 0.56° apart. The default **θ step (deg)** of 0.05°
draws them smoothly. At 0.5°, the quick button for a fast look, the fringes alias: you see the
envelope but not the fringes. Set the step back to 0.05 in the *Maker Fringes Scan Range* when you
need the individual fringes, and give the run about a minute.

**Why does my chosen material show "SHG ≈ 0 (symmetry-forbidden)"?**
Because its point group is centrosymmetric or isotropic (e.g. Air and Au ∞∞m, Al₂O₃ 6/mmm, Pt m3m,
"Blank linear" m3m). These groups sit in the *— Centrosymmetric (SHG-inactive) —* section of the
**Point group** dropdown; selecting any of them sets $d \equiv 0$ and collapses the *SHG Tensor
dᵢⱼ* group with the note "— not used: SHG-inactive point group (d ≡ 0)". SHG is forbidden by
symmetry for these, so the reflected/film signal is identically zero — the app states this clearly
instead of drawing a spurious curve. There is no separate "SHG active" switch: activity follows
the point group.

**What is the difference between the Full / JK / HH assumptions?**
They are different treatments of multiple reflections in the multilayer Maker sweep: **Full multiple
reflections (FMR)** keeps all multiply-reflected waves (with a forward/backward/standing sub-mode);
**Jerphagnon–Kurtz** drops multiple reflections (single pass); **Herman–Hayden** keeps them only for
the homogeneous $2\omega$ waves. See {doc}`ml_tab`.

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
It depends strongly on the method. The **field** (phase-resolved) method degrades gracefully: its
error tracks the noise level. The **intensity** (phase-less) method amplifies noise strongly at
typical conditioning, so treat its output on noisy data as an initial guess for *which components
are large*, not as a quantitative estimate. The numbers behind this are on {doc}`../validation`.

**The Update seems stuck on an analytical mode.**
It almost certainly is not — the analytical modes run a computer-algebra solve whose *first* run for
a configuration can take minutes (the progress bar says so). The result is cached: repeat Updates on
the same configuration return instantly.

**I entered a wavelength and got an amber note saying it is outside the material's index data.**
A material's dielectric tensors come from its built-in index data, which covers a set range.
Outside that range the tensors are held at the nearest tabulated value, and the note names the
material and its range so you know they are no longer wavelength-accurate there. A crystal from
the **Dispersive** group has a second case: λ can be inside its table while λ/2, where the second
harmonic reads the index, is below it. The note then says ε(2ω) is held at the table's first value
and names the wavelength from which both harmonics are covered. In a wavelength sweep the same thing
is said once per index table, with the range that table can answer.

**The note says ε(2ω) "is not physical at this wavelength", and my wavelength is inside the range.**
That is a different and more serious message. The permittivity at the second harmonic is read at
half the fundamental, so a wavelength well inside a material's data range can land on the
ultraviolet pole of its index data once it is halved. There ε(2ω) comes back as a lossless negative
or a runaway positive rather than an index. At a single wavelength the note says any result there
is not physical and the solve may fail; for the LiNbO₃ films at 1550 nm it fails below about
0.5 µm. A spectrum or Maker map that reaches that region refuses and names the window to use. A
Fresnel map is exempt, because it reads no second harmonic. Five palette films have such a floor;
the [usage page](../usage.md#materials-that-already-carry-dispersion) lists them.

**I ran a wavelength sweep, and the note says the permittivity does not move with wavelength.**
Most of the palette is defined at one wavelength, as the original notebooks defined it, so its
linear optics stay put across a sweep. On a single interface the sweep runs and plots a flat line.
On a layer stack the curve still moves, but only through each layer's optical thickness, so it is a
thickness sweep rather than a spectrum ({doc}`../conventions` explains why). The note appears after
**Update**, says so, and tells you what to pick instead. For a crystal with a table-backed version
it names that version, for example "KTP (dispersive) 0.43-3.54 um" for KTP (100); otherwise it
suggests a crystal from the **Dispersive** group. On the
SHAARP.si tab that group follows the case studies, above your own saved materials; on the SHAARP.ml
tab it follows the single-film palette.

**The note says "Only part of this stack disperses".**
One layer of your stack has an index that is a single number, so it stays at that value while the
other layers move with wavelength. In the Quartz + Au preset that layer is the Au coating. The
spectrum carries the other layers' dispersion but not that layer's. See {doc}`ml_tab`.

**My stack's spectrum has sharp spikes, and the note says the curve is "undersampled".**
A stack's interference fringes run along the wavelength axis, closer together the thicker the
layer. When the λ step is too coarse for those fringes, the curve samples them unevenly and shows
spikes that are not spectral features. The note names the layer, estimates the fringe spacing, and
suggests a step. The Quartz + Au preset, with its 121.2 µm quartz plate, needs a far finer step than
the default: narrow the range and use the suggested step. See {doc}`ml_tab`.

**How accurate is SHAARP.py?**
Every solver is checked against the published equations and the reference output of the original
SHAARP packages, and the published figures of both papers are reproduced through the same compute
path the app uses (see {doc}`../references`). The evidence, with tolerances, is on
{doc}`../validation`.

**Can I script this instead of using the GUI?**
Yes — see {doc}`../usage` and the {doc}`../api/index`. At one wavelength the GUI's **Update**
calls `compute_si_gui_result` and `compute_ml_gui_result`, which you can call yourself with the
same arguments the controls take; a wavelength sweep calls `run_si_spectrum`, `run_ml_spectrum` or
`run_spectral_map`.
