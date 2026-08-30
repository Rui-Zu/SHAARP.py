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

defaultInputPath = SHAARPPaths`Ref["maker_mathematica_inputs_v1.json"];
defaultOutputPath = SHAARPPaths`Ref["maker_fringes_reference_v1.json"];
defaultDiagnosticPath = SHAARPPaths`Ref["maker_fringes_reference_diagnostics.txt"];
inputPath = If[ValueQ[SHAARPPathOverrides`MakerInputPath], SHAARPPathOverrides`MakerInputPath, defaultInputPath];
outputPath = If[ValueQ[SHAARPPathOverrides`MakerOutputPath], SHAARPPathOverrides`MakerOutputPath, defaultOutputPath];
diagnosticPath = If[ValueQ[SHAARPPathOverrides`MakerDiagnosticPath], SHAARPPathOverrides`MakerDiagnosticPath, defaultDiagnosticPath];
inputPayload = Import[inputPath, "RawJSON"];

c0 = 1;
\[Mu]0 = 1;
\[CurlyEpsilon]0 = 1;
debug = False;
flagLinearSolve = False;
flagNSolve = True;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
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

safeMFDetail[mats_, thetatemp_, assumption_:0] := Module[
  {wIncLocal, wavLLocal, wavNLLocal, tmpw1Local, tmpw2Local, tEboLocal, tEruiLocal},
  wIncLocal = setwInc[\[Omega]0, N[(thetatemp/180) Pi], \[CapitalDelta]\[Delta], \[CurlyPhi], n0];
  extWave @ wIncLocal;
  If[assumption == 0,
    flagJK = False;
    flagHH = False;
    {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal];
    ,
    If[assumption == 1,
      flagBackward = False;
      flagStandingWave = False;
      flagJK = True;
      flagHH = False;
      {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal, True];
      ,
      If[assumption == 2,
        flagBackward = False;
        flagStandingWave = False;
        flagJK = False;
        flagHH = True;
        {wavLLocal, wavNLLocal} = f4NL[mats, wIncLocal, True];
      ];
    ];
  ];
  {tmpw1Local, tmpw2Local} = wavNLLocal[[2]];
  tEboLocal = (safeWaveA[tmpw2Local] + safeWaveA[tmpw1Local])[[1 ;; 2]];
  tEruiLocal = {{Cos[\[Psi]], Sin[\[Psi]]}, {-Sin[\[Psi]], Cos[\[Psi]]}} . tEboLocal;
  <|
    "theta" -> thetatemp,
    "tEboLocal" -> tEboLocal,
    "transmitted_two_omega_jones_sp" -> {tEboLocal[[2]], tEboLocal[[1]]},
    "parallel_analyzer_amplitude" -> tEruiLocal[[1]],
    "perpendicular_analyzer_amplitude" -> tEruiLocal[[2]],
    "MFRow" -> {
      thetatemp,
      tEruiLocal[[1]]*Conjugate[tEruiLocal[[1]]],
      tEruiLocal[[2]]*Conjugate[tEruiLocal[[2]]]
    }
  |>
];

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

makeCase[case_Association] := Module[
  {mats, thetaValues, mfDetails, mfList, listMFpara, listMFperp, caseOmega0, casePhi, casePsi, caseDelta, caseN0, symbolicAtoms},
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
  (* FMR sub-options (mrassumption=0): winhAssumption 0/1/2 -> backward / standing-wave flags,
     exactly as the original SHAARP.ml GUI maps them. winhAssumption=0 keeps the prior behavior
     (flagBackward=False), so existing winh=0 references are unchanged. *)
  Which[
    winhAssumption == 1, flagBackward = True; flagStandingWave = False,
    winhAssumption == 2, flagBackward = True; flagStandingWave = True,
    True, flagBackward = False; flagStandingWave = False
  ];
  flagJK = False;
  flagHH = False;

  thetaValues = N[case["theta_deg"], 17];
  mfDetails = (safeMFDetail[mats, #, case["mrassumption"]] &) /@ thetaValues;
  mfList = (#["MFRow"] &) /@ mfDetails;
  symbolicAtoms = DeleteDuplicates[Cases[mfList, s_Symbol /; Context[s] == "Global`", Infinity]];
  If[Length[symbolicAtoms] > 0,
    Return[
      <|
        "id" -> case["id"],
        "error" -> "MFList contains unresolved Global symbols",
        "symbol_count" -> Length[symbolicAtoms],
        "symbols" -> ToString[symbolicAtoms, InputForm],
        "mfListInputForm" -> ToString[mfList, InputForm]
      |>
    ]
  ];
  listMFpara = Re[mfList[[All, {1, 2}]]];
  listMFperp = Re[mfList[[All, {1, 3}]]];
  <|
    "id" -> case["id"],
    "inputs" -> <|
      "theta_deg" -> thetaValues,
      "phi_deg" -> case["phi_deg"],
      "psi_deg" -> case["psi_deg"],
      "ellipticity_deg" -> case["ellipticity_deg"],
      "mrassumption" -> case["mrassumption"],
      "winhAssumption" -> case["winhAssumption"]
    |>,
    "batch_export_notes" -> <|
      "raw_MF_empty_wave_diagnostic" -> "Original MF can attempt tmpw[A] on an empty wave slot such as {}[[1]] in FMR/no-backward cases.",
      "empty_wave_guard" -> "safeMF uses SHAARP.ml f4NL values but contributes zero for non-Association wave slots before summing transmitted A fields.",
      "diagnostic_file" -> "maker_fringes_reference_errors.json and wave_extension_diagnostics.json"
    |>,
    "mathematica_outputs" -> <|
      "MFList" -> jsonValue[mfList],
      "listMFpara" -> jsonValue[listMFpara],
      "listMFperp" -> jsonValue[listMFperp],
      "parallel_analyzer_amplitude" -> jsonValue[(#["parallel_analyzer_amplitude"] &) /@ mfDetails],
      "perpendicular_analyzer_amplitude" -> jsonValue[(#["perpendicular_analyzer_amplitude"] &) /@ mfDetails],
      "transmitted_two_omega_jones_sp" -> jsonValue[(#["transmitted_two_omega_jones_sp"] &) /@ mfDetails]
    |>
  |>
];

DeleteFile[diagnosticPath] /; FileExistsQ[diagnosticPath];
PutAppend["Starting Maker export with " <> ToString[Length[inputPayload["cases"]]] <> " cases", diagnosticPath];
cases = Table[
  PutAppend["Running case " <> ToString[i] <> ": " <> inputPayload["cases"][[i]]["id"], diagnosticPath];
  result = makeCase[inputPayload["cases"][[i]]];
  PutAppend["Finished case " <> ToString[i] <> " keys: " <> ToString[Keys[result], InputForm], diagnosticPath];
  result,
  {i, Length[inputPayload["cases"]]}
];

errorCases = Select[cases, KeyExistsQ[#, "error"] &];
If[Length[errorCases] > 0,
  Export[
    SHAARPPaths`Ref["maker_fringes_reference_errors.json"],
    <|
      "source" -> "Mathematica SHAARP.ml Maker export diagnostics",
      "status" -> "mathematica_reference_export_failed",
      "wolfram_version" -> $Version,
      "errors" -> errorCases
    |>,
    "JSON"
  ];
  PutAppend["Aborting Maker export because unresolved symbols were found.", diagnosticPath];
  Quit[1]
];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb Maker MFList/listMFpara/listMFperp reference values",
    "status" -> "mathematica_reference_exported_with_documented_empty_wave_guard",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "input_manifest" -> inputPath,
    "functionDownValues" -> <|"MF" -> Length[DownValues[MF]], "safeMFDetail" -> Length[DownValues[safeMFDetail]], "f4NL" -> Length[DownValues[f4NL]], "extWave" -> Length[DownValues[extWave]]|>,
    "batch_export_notes" -> {
      "Raw Mathematica MF failed for these FMR/no-backward cases because one transmitted nonlinear wave slot can be non-Association {}[[1]], leaving unresolved A.",
      "safeMF mirrors the original MF formula but treats non-Association transmitted wave slots as zero contribution.",
      "This is a documented batch-export workaround pending review against interactive GUI behavior."
    },
    "case_count" -> Length[cases],
    "suites" -> {
      <|
        "suite_id" -> "maker_fringes_workflow",
        "items" -> cases
      |>
    }
  |>,
  "JSON"
];
Quit[]
