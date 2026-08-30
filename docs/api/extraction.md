# d-tensor extraction

Recover the nonlinear $d_{ij}$ tensor from a reflected and/or transmitted SHG polarimetry scan — the
experimental inverse problem. Returns the recovered tensor together with identifiability /
conditioning diagnostics so you can see which components are well-determined.

```{warning}
**Noise robustness differs sharply between the two methods** (seeded Monte-Carlo
characterization: `benchmarks/dextraction_noise_benchmark.py`). With multiplicative intensity
noise of level $\sigma$:

- `method="field"` (phase-resolved amplitudes) degrades gracefully — median component error
  ≈ $\sigma$ on a dense multi-geometry scan;
- `method="intensity"` (phase-less Gram + rank-1) amplifies noise catastrophically at typical
  design conditioning ($\kappa \sim 10^4$–$10^5$): tens-of-percent errors at $\sigma = 2\%$. On
  noisy data treat its output as an initial guess for *which components are large*, not as a
  quantitative estimate.

Both methods recover the exact tensor on noise-free data (what the release gate verifies); check
the result's `condition_number` before trusting a noisy fit.
```

```{eval-rst}
.. automodule:: shaarp.polarimetry_extraction
   :members:
   :undoc-members:
   :show-inheritance:
```
