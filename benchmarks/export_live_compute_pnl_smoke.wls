toComplexJSON[z_] := <|"real" -> N[Re[z], 17], "imag" -> N[Im[z], 17]|>;
toComplexJSONList[x_List] := Map[toComplexJSON, x, {-1}];

d = {
  {0.11 + 0.02 I, -0.05, 0.04 I, 0.18, -0.09 I, 0.07},
  {-0.03 I, 0.08, -0.06, 0.05 I, 0.04, -0.02},
  {0.13, -0.10 I, 0.16, -0.08, 0.06 I, -0.01}
};
e1 = {0.7 + 0.2 I, -0.3 + 0.11 I, 0.45 - 0.07 I};
e2 = {-0.2 + 0.31 I, 0.62 - 0.14 I, -0.51 + 0.19 I};
factor = 1.3 - 0.4 I;

products = {
  e1[[1]] e2[[1]],
  e1[[2]] e2[[2]],
  e1[[3]] e2[[3]],
  e1[[2]] e2[[3]] + e1[[3]] e2[[2]],
  e1[[1]] e2[[3]] + e1[[3]] e2[[1]],
  e1[[2]] e2[[1]] + e1[[1]] e2[[2]]
};
pnl = factor d . products;

payload = <|
  "source" -> "Live Wolfram Mathematica batch export",
  "status" -> "mathematica_reference_exported",
  "batch_execution" -> "file_export_confirmed",
  "wolfram_version" -> $Version,
  "cases" -> {
    <|
      "id" -> "live_compute_pnl_complex_smoke_001",
      "inputs" -> <|
        "d_voigt_lab" -> toComplexJSONList[d],
        "e1" -> toComplexJSONList[e1],
        "e2" -> toComplexJSONList[e2],
        "factor" -> toComplexJSON[factor]
      |>,
      "outputs" -> <|
        "pnl" -> toComplexJSONList[pnl]
      |>
    |>
  }
|>;

Export[
  FileNameJoin[{Directory[], "Python Ver", "shaarp_py", "benchmarks", "mathematica_reference", "wolfram_live_compute_pnl_smoke_v1.json"}],
  payload,
  "JSON"
];
Quit[];
