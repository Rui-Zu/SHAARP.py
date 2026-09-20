# Sample rotation

Turning the sample is not the same experiment as turning the light. When you rotate a crystal about
its surface normal, every tensor the solver works with turns too, so the refractive behaviour and
the nonlinear response both change together. This page derives what that rotation does, in the
conventions the package actually uses, so you can read a rotational-anisotropy pattern and know
where each part of it comes from.

The frames, the angles and the Voigt convention are defined in {doc}`conventions`. This page uses
them without restating them.

## The rotation

The sample azimuth $\psi_s$ is a rotation about the lab axis $L_3$, the surface normal, which
points into the sample. Write it as

$$R_z(\psi_s) = \begin{pmatrix} \cos\psi_s & -\sin\psi_s & 0 \\ \sin\psi_s & \cos\psi_s & 0 \\ 0 & 0 & 1\end{pmatrix}.$$

There are two senses in play and it is worth separating them once. A positive $\psi_s$ in the
solver is a right-handed rotation about that inward normal, which someone looking at the sample
from the beam side sees as clockwise. The app's polar scan is labelled the way an experimentalist
stands, so its counter-clockwise sweep is the negated angle. The scan direction control chooses
between the two, and nothing else in the package depends on that choice.

## What the orientation carries

A crystal's orientation is stored as the matrix $A$ whose rows are the crystal-physics axes
expressed in lab coordinates. Rotating the sample rotates each of those axes, so each row is
multiplied by $R_z$, which for the whole matrix reads

$$A(\psi_s) = A_0\,R_z(\psi_s)^{\mathsf T}.$$

That single line is the definition of a sample rotation in this package. Everything below follows
from it, and every mode that offers a sample azimuth applies exactly this.

## The dielectric tensor

The permittivity in the lab frame follows the rotation of the axes:

$$\varepsilon^{\text{lab}}(\psi_s) = R_z(\psi_s)\,\varepsilon^{\text{lab}}(0)\,R_z(\psi_s)^{\mathsf T}.$$

For an isotropic crystal, or a uniaxial one whose optic axis lies along the surface normal, this is
the identity: the permittivity does not notice the rotation at all. Then the refractive part of the
problem, the mode structure, the wavevectors, the Fresnel coefficients and the propagation phases,
is independent of $\psi_s$.

For anything else, a biaxial crystal or a tilted optic axis, the permittivity genuinely turns. The
eigenvalue problem that selects the propagating modes then turns with it, and there is no shortcut:
the modes have to be found again at each azimuth. This distinction is the one that decides which
route a calculation can take, and it comes back below.

## The nonlinear tensor

The second-order response is a rank-three tensor, so it carries three factors of the rotation.
Substituting $A(\psi_s)$ into the crystal-to-lab transformation and collecting terms gives the
lab-frame tensor as a rotation of the unrotated lab-frame one:

$$d^{\text{lab}}_{ijk}(\psi_s) = R_{z,im}\,R_{z,jn}\,R_{z,kp}\;d^{\text{lab}}_{mnp}(0).$$

Each entry is therefore a polynomial in $\cos\psi_s$ and $\sin\psi_s$ of degree at most three, which
is to say a combination of $\cos n\psi_s$ and $\sin n\psi_s$ for $n \le 3$. That is the origin of
the lobe count you see in a polar scan: the crystal's point group decides which harmonics survive,
and the surviving harmonics are what the detector traces out as the sample turns.

## Voigt bookkeeping

The nonlinear tensor is stored contracted, as a $3\times6$ matrix. The factor of two that the
contraction needs lives on the field product, not on the tensor, so the rotation above is an
ordinary tensor rotation with no extra factors to carry. Expanding to the full rank-three form,
rotating, and contracting again is exactly what the package does, and it is why a rotated tensor
can be read back in the same units as an unrotated one.

## What a positive azimuth means

A positive $\psi_s$ means one thing everywhere in the package: the rotation the orientation defines,
the one written above. That is worth stating explicitly because the opposite choice is easy to make
and almost impossible to see. A mirrored azimuth produces a pattern that is still smooth, still has
the right number of lobes, still has the right magnitude, and is simply reflected. Only a signed
comparison against a known rotation catches it, and a crystal with three-fold symmetry examined at
one hundred and twenty degrees cannot catch it at all.

## When the azimuth can stay symbolic

The nonlinear response is linear in every component of the nonlinear tensor. The boundary-value
problem is solved once per component, and the answer at a given azimuth is the sum of those
solutions weighted by that component's value.

When the permittivity does not turn, the whole linear problem is azimuth-independent and only the
weights change. A tensor whose entries are trigonometric in $\psi_s$ then produces a response that
is trigonometric in $\psi_s$, with no further solving. That is what lets a closed form carry the
azimuth as a symbol, and it is also what lets a numerical scan cost one solve per tensor component
rather than one per angle, however fine the angular step.

When the permittivity does turn, none of that holds. The mode structure changes with the azimuth,
the closed form has no counterpart, and the response is computed point by point. The app checks
which case it is in and takes the matching route.

## What the detector sees

At each azimuth the outgoing second-harmonic field is projected onto the analyzer. Writing the
field in its parallel and perpendicular components relative to the plane of incidence, the analyzer
at $\psi$ selects

$$\begin{pmatrix}I_\parallel \\ I_\perp\end{pmatrix} \propto \left|\begin{pmatrix}\cos\psi & -\sin\psi \\ \sin\psi & \cos\psi\end{pmatrix}\begin{pmatrix}E_p \\ E_s\end{pmatrix}\right|^2$$

for the reflected side. The transmitted side uses the transpose of that matrix, because the two
sides are referred to opposite propagation directions. Holding the polarizer and analyzer fixed
while the sample turns is what makes the resulting polar plot a property of the crystal rather than
of the optics, which is why the lobe count reads its rotational symmetry directly.

Away from normal incidence, turning the sample is not equivalent to turning the polarizer and
analyzer together. At normal incidence the two coincide, and they part company immediately as the
incidence angle grows.

## Where the azimuth appears

| Method | Mode | The azimuth is |
|---|---|---|
| Single interface | SHG Simulation | a fixed setting |
| Single interface | Partial Analytical | a fixed setting |
| Single interface | Full Analytical | a fixed setting |
| Multilayer | SHG Simulation | the scanned variable, or a fixed setting |
| Multilayer | Maker Fringes | a fixed setting, with the incidence angle scanned |
| Multilayer | Fresnel Coefficients | a fixed setting; it changes nothing unless a layer is optically anisotropic, and the result says which case it is |
| Multilayer | Partial Analytical | the scanned variable, carried symbolically where the permittivity allows, or a fixed setting |

Both the single-interface and the multilayer methods offer it, and a positive azimuth means the
same physical rotation in both.
