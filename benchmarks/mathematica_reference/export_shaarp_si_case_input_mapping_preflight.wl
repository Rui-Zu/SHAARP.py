(* --- Path resolution ---------------------------------------------------------------
   SHAARP_REF_DIR is this checkout's benchmarks/mathematica_reference, and defaults to the
   directory containing this script.
   SHAARP_ML_DIR is a local checkout of github.com/bzw133/SHAARP.ml, which provides setup.nb.
   SHAARP_SI_DIR is a local checkout of github.com/Rui-Zu/SHAARP, which provides SHAARP_V1.03.
   The environment is consulted FIRST: the batch runner executes a COPY of this script from a
   temporary directory, where $InputFileName would point at the copy rather than the checkout. *)
SHAARPPaths`RefDir = With[{env = Environment["SHAARP_REF_DIR"]},
  Which[
    StringQ[env] && DirectoryQ[env], env,
    StringQ[$InputFileName] && $InputFileName =!= "", DirectoryName[$InputFileName],
    True, Directory[]]];
SHAARPPaths`Ref[name_String] := FileNameJoin[{SHAARPPaths`RefDir, name}];
(* RefDir is <repo>/benchmarks/mathematica_reference, so the repository root is two levels up. *)
SHAARPPaths`Repo[rel_String] :=
  FileNameJoin[{ParentDirectory[ParentDirectory[SHAARPPaths`RefDir]], rel}];
SHAARPPaths`ExternalDir[var_String, repo_String] :=
  With[{env = Environment[var]},
    If[StringQ[env] && DirectoryQ[env],
      env,
      (Print["Set " <> var <> " to a local checkout of " <> repo <>
         ": this script re-exports a reference from the original Mathematica package, " <>
         "which SHAARP.py does not vendor."]; Quit[2])]];
SHAARPPaths`MLDir[] := SHAARPPaths`ExternalDir["SHAARP_ML_DIR", "github.com/bzw133/SHAARP.ml"];
SHAARPPaths`SIDir[] := SHAARPPaths`ExternalDir["SHAARP_SI_DIR", "github.com/Rui-Zu/SHAARP"];
SHAARPPaths`ML[rel_String] := FileNameJoin[{SHAARPPaths`MLDir[], rel}];
SHAARPPaths`SI[rel_String] := FileNameJoin[{SHAARPPaths`SIDir[], rel}];
(* ------------------------------------------------------------------------------------ *)

inputPath = SHAARPPaths`Ref["shaarp_si_case_input_mapping_v1.json"];
outputPath = SHAARPPaths`Ref["shaarp_si_case_input_mapping_preflight_v1.json"];

payload = Import[inputPath, "RawJSON"];

toComplex[x_Association] /; KeyExistsQ[x, "real"] := N[x["real"] + I*x["imag"], 17];
toComplex[x_List] := toComplex /@ x;
toComplex[x_] := x;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;

maxAbs[x_] := Max[Abs[Flatten[N[x]]]];

voigtToChi2[d_] := Module[{out, pairs},
  out = ConstantArray[0, {3, 3, 3}];
  pairs = <|
    1 -> {{1, 1}},
    2 -> {{2, 2}},
    3 -> {{3, 3}},
    4 -> {{2, 3}, {3, 2}},
    5 -> {{1, 3}, {3, 1}},
    6 -> {{1, 2}, {2, 1}}
  |>;
  Do[
    Do[
      Do[out[[i, pair[[1]], pair[[2]]]] = d[[i, col]], {pair, pairs[col]}],
      {col, 1, 6}
    ],
    {i, 1, 3}
  ];
  out
];

chi2ToVoigt[chi_] := Transpose[{
  chi[[All, 1, 1]],
  chi[[All, 2, 2]],
  chi[[All, 3, 3]],
  0.5*(chi[[All, 2, 3]] + chi[[All, 3, 2]]),
  0.5*(chi[[All, 1, 3]] + chi[[All, 3, 1]]),
  0.5*(chi[[All, 1, 2]] + chi[[All, 2, 1]])
}];

crystalToLabD[dCrystal_, a_] := Module[{chiCrystal, chiLab},
  chiCrystal = voigtToChi2[dCrystal];
  chiLab = Table[
    Sum[
      a[[i, l]]*a[[j, m]]*a[[k, n]]*chiCrystal[[l, m, n]],
      {l, 1, 3}, {m, 1, 3}, {n, 1, 3}
    ],
    {i, 1, 3}, {j, 1, 3}, {k, 1, 3}
  ];
  chi2ToVoigt[chiLab]
];

caseRecord[case_] := Module[
  {vars, expected, a, epsW, eps2W, dOld, epsWLab, eps2WLab, dLab, epsWErr, eps2WErr, dErr},
  vars = case["mathematica_variables"];
  expected = case["expected_python_lab_inputs"];
  a = toComplex[vars["a"]];
  epsW = toComplex[vars["\\[CurlyEpsilon]\\[Omega]Cry"]];
  eps2W = toComplex[vars["\\[CurlyEpsilon]2\\[Omega]Cry"]];
  dOld = toComplex[vars["dold"]];
  epsWLab = toComplex[expected["epsilon_omega_lab"]];
  eps2WLab = toComplex[expected["epsilon_2omega_lab"]];
  dLab = toComplex[expected["d_voigt_lab"]];
  epsWErr = maxAbs[a.epsW.Transpose[a] - epsWLab];
  eps2WErr = maxAbs[a.eps2W.Transpose[a] - eps2WLab];
  dErr = maxAbs[crystalToLabD[dOld, a] - dLab];
  <|
    "id" -> case["id"],
    "max_abs_epsilon_omega_lab_error" -> N[epsWErr, 17],
    "max_abs_epsilon_2omega_lab_error" -> N[eps2WErr, 17],
    "max_abs_d_voigt_lab_error" -> N[dErr, 17]
  |>
];

records = caseRecord /@ payload["cases"];

Export[
  outputPath,
  <|
    "source" -> "Mathematica preflight for SHAARP.si case input mapping",
    "status" -> "mathematica_preflight_exported_not_shaarp_stage_values",
    "validation_claim" -> False,
    "wolfram_version" -> $Version,
    "input_path" -> inputPath,
    "case_count" -> Length[records],
    "max_abs_epsilon_omega_lab_error" -> Max[records[[All, "max_abs_epsilon_omega_lab_error"]]],
    "max_abs_epsilon_2omega_lab_error" -> Max[records[[All, "max_abs_epsilon_2omega_lab_error"]]],
    "max_abs_d_voigt_lab_error" -> Max[records[[All, "max_abs_d_voigt_lab_error"]]],
    "cases" -> records
  |>,
  "JSON"
];

Quit[]
