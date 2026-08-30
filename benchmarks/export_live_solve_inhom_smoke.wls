toComplexJSON[z_] := <|"real" -> N[Re[z], 17], "imag" -> N[Im[z], 17]|>;
toComplexJSONList[x_List] := Map[toComplexJSON, x, {-1}];

crossMatrix[v_] := {
  {0, -v[[3]], v[[2]]},
  {v[[3]], 0, -v[[1]]},
  {-v[[2]], v[[1]], 0}
};

epsilon = {
  {2.41 + 0.07 I, 0.06 - 0.03 I, -0.04 + 0.02 I},
  {0.05 + 0.04 I, 2.88 + 0.05 I, 0.03 - 0.01 I},
  {-0.02 + 0.03 I, 0.04 + 0.02 I, 3.32 + 0.06 I}
};
k = {0.42 + 0.03 I, -0.17 + 0.02 I, 1.63 + 0.05 I};
polarization = {0.31 - 0.12 I, -0.24 + 0.09 I, 0.18 + 0.21 I};
omega = 2.0;
mu = 1.0;
eps0 = 1.0;

kx = crossMatrix[k];
operator = -(kx . kx);
alpha = omega^2 mu eps0;
system = operator - alpha epsilon;
electric = LinearSolve[system, alpha polarization];
residual = operator . electric - alpha (epsilon . electric + polarization);

payload = <|
  "source" -> "Live Wolfram Mathematica batch export",
  "status" -> "mathematica_reference_exported",
  "batch_execution" -> "file_export_confirmed",
  "wolfram_version" -> $Version,
  "field_convention" -> "exp(i(omega*t - k.r))",
  "cases" -> {
    <|
      "id" -> "live_solve_inhom_complex_nondiagonal_smoke_001",
      "inputs" -> <|
        "epsilon_2omega_lab" -> toComplexJSONList[epsilon],
        "k" -> toComplexJSONList[k],
        "polarization" -> toComplexJSONList[polarization],
        "omega" -> toComplexJSON[omega],
        "mu" -> toComplexJSON[mu],
        "eps0" -> toComplexJSON[eps0]
      |>,
      "outputs" -> <|
        "electric" -> toComplexJSONList[electric],
        "residual" -> toComplexJSONList[residual]
      |>
    |>
  }
|>;

Export[
  FileNameJoin[{Directory[], "Python Ver", "shaarp_py", "benchmarks", "mathematica_reference", "wolfram_live_solve_inhom_smoke_v1.json"}],
  payload,
  "JSON"
];
Quit[];
