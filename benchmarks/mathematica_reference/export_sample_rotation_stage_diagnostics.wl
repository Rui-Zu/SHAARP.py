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

SetDirectory[SHAARPPaths`MLDir[]];
nb = Import[SHAARPPaths`ML["setup.nb"], "NB"];
cells = Cases[nb, Cell[box_, "Input", ___] :> box, Infinity];
SHAARPReferenceLoader`inputCellCount = Length[cells];
held = ToExpression[cells, StandardForm, HoldComplete];
Scan[ReleaseHold, held];

inputPath = SHAARPPaths`Ref["sample_rotation_mathematica_inputs_v1.json"];
outputPath = SHAARPPaths`Ref["sample_rotation_stage_diagnostics_v1.json"];
inputPayload = Import[inputPath, "RawJSON"];
Print["Loaded SampleRotate stage input cases: ", Length[inputPayload["cases"]]];

c0 = 1;
\[Mu]0 = 1;
\[CurlyEpsilon]0 = 1;
debug = False;
flagLinearSolve = False;
flagNSolve = True;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_Association] := AssociationMap[jsonValue, x];
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
jsonValue[x_] := ToString[x, InputForm];

fromJsonComplex[x_Association] := N[x["real"] + I*x["imag"], 17];
fromJsonComplex[x_List] := fromJsonComplex /@ x;
fromJsonComplex[x_?NumericQ] := N[x, 17];
safeReal[x_] := Re[N[x, 17]];

safeWaveA[wav_] := Module[{copy = wav},
  If[Head[copy] =!= Association,
    Return[ConstantArray[0, 3]]
  ];
  If[!TrueQ[KeyExistsQ[copy, A]],
    extWave @ copy
  ];
  copy[A]
];

waveRecord[label_, wav_] := Module[{copy = wav, symbols},
  If[Head[copy] =!= Association,
    Return[<|
      "label" -> label,
      "present" -> False,
      "head" -> ToString[Head[copy], InputForm],
      "input_form" -> ToString[copy, InputForm]
    |>]
  ];
  If[!TrueQ[KeyExistsQ[copy, A]], extWave @ copy];
  symbols = DeleteDuplicates[Cases[copy, s_Symbol /; Context[s] == "Global`", Infinity]];
  <|
    "label" -> label,
    "present" -> True,
    "omega" -> jsonValue[copy[\[Omega]]],
    "theta" -> jsonValue[copy[\[Theta]]],
    "k" -> jsonValue[copy[k]],
    "k0" -> jsonValue[copy[k0]],
    "electric" -> jsonValue[copy[EE]],
    "A" -> jsonValue[copy[A]],
    "H" -> jsonValue[copy[H]],
    "symbols" -> ToString[symbols, InputForm]
  |>
];

waveListRecords[prefix_, waves_] := MapIndexed[waveRecord[prefix <> "_" <> ToString[#2[[1]]], #1] &, waves];

makeMaterial[layer_Association] := Module[
  {orientationRows, qc, epsW, eps2W, d, hValue, pgValue, mat},
  orientationRows = safeReal /@ fromJsonComplex[layer["orientation_rows"]];
  qc = safeReal /@ fromJsonComplex[layer["qc"]];
  epsW = fromJsonComplex[layer["epsilon_omega_crystal"]];
  eps2W = fromJsonComplex[layer["epsilon_2omega_crystal"]];
  d = fromJsonComplex[layer["d_voigt_crystal"]];
  hValue = If[layer["thickness_um"] === Null, 0., N[layer["thickness_um"], 17]];
  pgValue = {layer["point_group"], ConstantArray[0, {3, 6}], layer["nonlinear_flag"]};
  mat = setMater[{
      layer["material_name"],
      {2, {{0, 0, 1}, {0, 1, 0}}, orientationRows},
      pgValue,
      layer["latcon"],
      epsW,
      eps2W,
      d,
      hValue,
      qc
    }];
  extMater[mat];
  mat
];

SetAttributes[rotateMatsInPlace, HoldFirst];
rotateMatsInPlace[mats_, rotStepDeg_, totalmats_] := Module[{i, matstemp, rot},
  rot = N[{{Cos[rotStepDeg Degree], -Sin[rotStepDeg Degree], 0},
      {Sin[rotStepDeg Degree], Cos[rotStepDeg Degree], 0},
      {0, 0, 1}}, 17];
  For[i = 2, i <= totalmats + 1, i++,
    matstemp = mats[[i]];
    If[matstemp[[Key[orientation]]][[1]] == 1,
      matstemp[[Key[orientation]]][[1]] = 2
    ];
    matstemp[[Key[orientation]]][[3]][[1]] = Chop[N[rot . matstemp[[Key[orientation]]][[3]][[1]], 17]];
    matstemp[[Key[orientation]]][[3]][[2]] = Chop[N[rot . matstemp[[Key[orientation]]][[3]][[2]], 17]];
    matstemp[[Key[orientation]]][[3]][[3]] = Chop[N[rot . matstemp[[Key[orientation]]][[3]][[3]], 17]];
    hklConvert[matstemp];
    extMater[matstemp];
    mats[[i]] = matstemp;
  ];
];

layerOrientationRecords[mats_, totalmats_] := Table[
  <|
    "mathematica_layer_index" -> i,
    "orientation_rows" -> jsonValue[mats[[i]][[Key[orientation]]][[3]]],
    "qc" -> jsonValue[mats[[i]][[Key[QC]]]]
  |>,
  {i, 2, totalmats + 1}
];

SetAttributes[makePoint, HoldFirst];
makePoint[mats_, thetaDeg_, rotateDeg_, rotateStepDeg_, assumption_, totalmats_, delta_, phi_, psi_] := Module[
  {wIncLocal, wavLLocal, wavNLLocal, r1, r2, t1, t2, rEbo, tEbo, rErui, tErui},
  If[rotateDeg != 0,
    rotateMatsInPlace[mats, rotateStepDeg, totalmats]
  ];
  \[CapitalDelta]\[Delta] = delta;
  \[CurlyPhi] = phi;
  \[Psi] = psi;
  wIncLocal = setwInc[\[Omega]0, N[(thetaDeg/180) Pi, 17], \[CapitalDelta]\[Delta], \[CurlyPhi], n0];
  extWave @ wIncLocal;
  If[assumption == 0,
    flagJK = False; flagHH = False;
    {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal];,
    If[assumption == 1,
      flagBackward = False; flagStandingWave = False; flagJK = True; flagHH = False;
      {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal, True];,
      flagBackward = False; flagStandingWave = False; flagJK = False; flagHH = True;
      {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal, True];
    ];
  ];
  {r1, r2} = wavNLLocal[[1]];
  {t1, t2} = wavNLLocal[[2]];
  rEbo = (safeWaveA[r2] + safeWaveA[r1])[[1 ;; 2]];
  tEbo = (safeWaveA[t2] + safeWaveA[t1])[[1 ;; 2]];
  rErui = {{Cos[psi], -Sin[psi]}, {Sin[psi], Cos[psi]}} . rEbo;
  tErui = {{Cos[psi], Sin[psi]}, {-Sin[psi], Cos[psi]}} . tEbo;
  <|
    "sample_rotation_deg" -> rotateDeg,
    "rotation_step_deg" -> rotateStepDeg,
    "incident" -> waveRecord["incident", wIncLocal],
    "layer_orientations_after_rotation" -> layerOrientationRecords[mats, totalmats],
    "linear_reflected" -> waveListRecords["linear_reflected", wavLLocal[[1]]],
    "linear_transmitted" -> waveListRecords["linear_transmitted", wavLLocal[[2]]],
    "nonlinear_reflected" -> waveListRecords["nonlinear_reflected", wavNLLocal[[1]]],
    "nonlinear_transmitted" -> waveListRecords["nonlinear_transmitted", wavNLLocal[[2]]],
    "sample_rotate_amplitudes" -> <|
      "reflected_parallel" -> jsonValue[rErui[[1]]],
      "reflected_perpendicular" -> jsonValue[rErui[[2]]],
      "transmitted_parallel" -> jsonValue[tErui[[1]]],
      "transmitted_perpendicular" -> jsonValue[tErui[[2]]],
      "reflected_wave_frame_xy_sum" -> jsonValue[rEbo],
      "transmitted_wave_frame_xy_sum" -> jsonValue[tEbo]
    |>,
    "sample_rotate_intensities" -> jsonValue[{
      Re[rErui[[1]] Conjugate[rErui[[1]]]],
      Re[rErui[[2]] Conjugate[rErui[[2]]]],
      Re[tErui[[1]] Conjugate[tErui[[1]]]],
      Re[tErui[[2]] Conjugate[tErui[[2]]]]
    }]
  |>
];

makeCase[case_Association] := Module[
  {mats, rotations, theta, delta, phi, psi, rows, totalmats},
  mats = makeMaterial /@ case["layers"];
  totalmats = Length[mats] - 2;
  rotations = N[case["sample_rotation_deg"], 17];
  theta = N[case["theta_deg"], 17];
  delta = N[case["ellipticity_deg"] Degree, 17];
  phi = N[case["phi_deg"] Degree, 17];
  psi = N[case["psi_deg"] Degree, 17];
  \[Omega]0 = N[case["omega0"], 17];
  n0 = Sqrt[Mean[Eigenvalues[mats[[1]][\[CurlyEpsilon]\[Omega]C]]]];
  winhAssumption = case["winhAssumption"];
  flagBackward = False;
  flagStandingWave = False;
  rows = Table[
    Module[{rot = rotations[[idx]], step},
      step = If[idx == 1, rot, rot - rotations[[idx - 1]]];
      makePoint[mats, theta, rot, step, case["mrassumption"], totalmats, delta, phi, psi]
    ],
    {idx, Length[rotations]}
  ];
  <|
    "id" -> case["id"],
    "source_case_id" -> case["source_case_id"],
    "theta_deg" -> theta,
    "phi_deg" -> case["phi_deg"],
    "psi_deg" -> case["psi_deg"],
    "ellipticity_deg" -> case["ellipticity_deg"],
    "mrassumption" -> case["mrassumption"],
    "winhAssumption" -> case["winhAssumption"],
    "points" -> rows
  |>
];

cases = makeCase /@ inputPayload["cases"];
Print["Computed SampleRotate stage cases: ", Length[cases]];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb SampleRotate stage diagnostics",
    "status" -> "mathematica_sample_rotation_stage_diagnostics_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "input_manifest" -> inputPath,
    "functionDownValues" -> <|"SampleRotate" -> Length[DownValues[SampleRotate]], "f4NL" -> Length[DownValues[f4NL]], "solveInhom" -> Length[DownValues[solveInhom]], "solveFresnelN" -> Length[DownValues[solveFresnelN]]|>,
    "case_count" -> Length[cases],
    "cases" -> cases
  |>,
  "JSON"
];
Print["Wrote SampleRotate stage diagnostics: ", outputPath];
Quit[]
