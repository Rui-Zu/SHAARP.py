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
Because it is under-sampled. That heterostructure's SHG-active layer is a 121.2 µm z-cut quartz
plate, whose Maker fringes in incidence angle are very dense, and the default 0.5° step cannot
resolve them; the envelope and the peak position are right, the fine structure is not. Reduce the
angle step in **Maker Fringes** if you need the individual fringes, and expect the run to take
proportionally longer.


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

**I entered a wavelength and got an amber note about a "tabulated range".**
Case-study dielectric tensors are interpolated from the published case studies' dispersion
grids. Outside a grid the tensors clamp to the nearest tabulated value; the note names the affected
material and its valid range so you know the tensors are no longer wavelength-accurate there.

**How accurate is SHAARP.py?**
Every solver is checked against the published equations and the reference output of the original
SHAARP packages, and the published figures of both papers are reproduced through the same compute
path the app uses (see {doc}`../references`). The evidence, with tolerances, is on
{doc}`../validation`.

**Can I script this instead of using the GUI?**
Yes — see {doc}`../usage` and the {doc}`../api/index`. The GUI's **Update** calls the same
`run_*` functions the API exposes.
