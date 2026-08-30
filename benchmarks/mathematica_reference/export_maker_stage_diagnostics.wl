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

inputPath = SHAARPPaths`Ref["maker_mathematica_inputs_v1.json"];
outputPath = SHAARPPaths`Ref["maker_stage_diagnostics_v1.json"];
inputPayload = Import[inputPath, "RawJSON"];

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

makePoint[case_Association, thetaDeg_] := Module[
  {mats, wIncLocal, wavLLocal, wavNLLocal, caseOmega0, casePhi, casePsi, caseDelta},
  caseOmega0 = N[case["omega0"], 17];
  casePhi = N[case["phi_deg"] Degree, 17];
  casePsi = N[case["psi_deg"] Degree, 17];
  caseDelta = N[case["ellipticity_deg"] Degree, 17];
  mats = makeMaterial /@ case["layers"];

  \[Omega]0 = caseOmega0;
  \[CapitalDelta]\[Delta] = caseDelta;
  \[CurlyPhi] = casePhi;
  \[Psi] = casePsi;
  n0 = Sqrt[Mean[Eigenvalues[mats[[1]][\[CurlyEpsilon]\[Omega]C]]]];
  winhAssumption = case["winhAssumption"];
  flagBackward = False;
  flagStandingWave = False;
  flagJK = False;
  flagHH = False;

  wIncLocal = setwInc[\[Omega]0, N[(thetaDeg/180) Pi], \[CapitalDelta]\[Delta], \[CurlyPhi], n0];
  extWave @ wIncLocal;
  If[case["mrassumption"] == 0,
    flagJK = False;
    flagHH = False;
    {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal];
    ,
    If[case["mrassumption"] == 1,
      flagBackward = False;
      flagStandingWave = False;
      flagJK = True;
      flagHH = False;
      {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal, True];
      ,
      flagBackward = False;
      flagStandingWave = False;
      flagJK = False;
      flagHH = True;
      {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal, True];
    ];
  ];
  <|
    "theta_deg" -> thetaDeg,
    "incident" -> waveRecord["incident", wIncLocal],
    "linear_reflected" -> waveListRecords["linear_reflected", wavLLocal[[1]]],
    "linear_transmitted" -> waveListRecords["linear_transmitted", wavLLocal[[2]]],
    "nonlinear_reflected" -> waveListRecords["nonlinear_reflected", wavNLLocal[[1]]],
    "nonlinear_transmitted" -> waveListRecords["nonlinear_transmitted", wavNLLocal[[2]]],
    "backward_homogeneous_by_layer" -> MapIndexed[waveListRecords["backward_homogeneous_layer_" <> ToString[#2[[1]]], #1] &, wavNLLocal[[3]]],
    "forward_homogeneous_by_layer" -> MapIndexed[waveListRecords["forward_homogeneous_layer_" <> ToString[#2[[1]]], #1] &, wavNLLocal[[4]]],
    "inhomogeneous_by_layer" -> MapIndexed[waveListRecords["inhomogeneous_layer_" <> ToString[#2[[1]]], #1] &, wavNLLocal[[5]]]
  |>
];

makeCase[case_Association] := <|
  "id" -> case["id"],
  "mrassumption" -> case["mrassumption"],
  "winhAssumption" -> case["winhAssumption"],
  "points" -> (makePoint[case, #] &) /@ N[case["theta_deg"], 17]
|>;

cases = makeCase /@ inputPayload["cases"];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb Maker f4NL stage diagnostics",
    "status" -> "mathematica_stage_diagnostics_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"f4NL" -> Length[DownValues[f4NL]], "f4L" -> Length[DownValues[f4L]], "solveSnell" -> Length[DownValues[solveSnell]], "solveInhom" -> Length[DownValues[solveInhom]], "solveFresnelN" -> Length[DownValues[solveFresnelN]]|>,
    "case_count" -> Length[cases],
    "cases" -> cases
  |>,
  "JSON"
];
Quit[]
