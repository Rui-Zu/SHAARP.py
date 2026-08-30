(* --- Path resolution ---------------------------------------------------------------
   SHAARP_REF_DIR is this checkout's benchmarks/mathematica_reference, and defaults to the
   directory containing this script.
   SHAARP_ML_DIR is a local checkout of github.com/Rui-Zu/SHAARP.ml, which provides setup.nb.
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
SHAARPPaths`MLDir[] := SHAARPPaths`ExternalDir["SHAARP_ML_DIR", "github.com/Rui-Zu/SHAARP.ml"];
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
outputPath = SHAARPPaths`Ref["fresnel_sweep_reference_v1.json"];
diagnosticPath = SHAARPPaths`Ref["fresnel_sweep_reference_diagnostics.txt"];
inputPayload = Import[inputPath, "RawJSON"];

c0 = 1;
\[Mu]0 = 1;
\[CurlyEpsilon]0 = 1;
debug = False;
flagLinearSolve = False;
flagNSolve = True;
noAnalytical = True;

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

safeFresnelRow[mats_, thetaDeg_, phiDeg_, deltaDeg_, assumption_:0] := Module[
  {wIncLocal, wavLLocal, tmpw1Local, tmpw2Local, rmpw1Local, rmpw2Local, rEtemp, tEtemp},
  \[CurlyPhi] = N[phiDeg Degree, 17];
  \[CapitalDelta]\[Delta] = N[deltaDeg Degree, 17];
  wIncLocal = setwInc[\[Omega]0, N[(thetaDeg/180) Pi], \[CapitalDelta]\[Delta], \[CurlyPhi], n0];
  extWave @ wIncLocal;
  If[assumption == 0,
    wavLLocal = f4L[mats, wIncLocal];
    ,
    If[assumption == 1 || assumption == 2,
      wavLLocal = f4L[mats, wIncLocal, True];
    ];
  ];
  {rmpw1Local, rmpw2Local} = wavLLocal[[1]];
  {tmpw1Local, tmpw2Local} = wavLLocal[[2]];
  rEtemp = (safeWaveA[rmpw1Local] + safeWaveA[rmpw2Local])[[1 ;; 2]];
  tEtemp = (safeWaveA[tmpw1Local] + safeWaveA[tmpw2Local])[[1 ;; 2]];
  N[
    {
      thetaDeg,
      rEtemp[[1]]*Conjugate[rEtemp[[1]]],
      rEtemp[[2]]*Conjugate[rEtemp[[2]]],
      tEtemp[[1]]*Conjugate[tEtemp[[1]]],
      tEtemp[[2]]*Conjugate[tEtemp[[2]]]
    },
    17
  ]
];

curveRows[mats_, thetaValues_, phiDeg_, columnIndex_, assumption_:0] := Module[
  {rows},
  rows = (safeFresnelRow[mats, #, phiDeg, 0, assumption] &) /@ thetaValues;
  N[Re[rows[[All, {1, columnIndex}]]], 17]
];

makeCase[case_Association] := Module[
  {mats, thetaValues, directRows, listFresnel, symbolicAtoms},
  mats = makeMaterial /@ case["layers"];
  \[Omega]0 = N[case["omega0"], 17];
  n0 = Sqrt[Mean[Eigenvalues[mats[[1]][\[CurlyEpsilon]\[Omega]C]]]];
  flagBackward = False;
  flagStandingWave = False;
  flagJK = False;
  flagHH = False;

  thetaValues = N[case["theta_deg"], 17];
  directRows = (safeFresnelRow[mats, #, case["phi_deg"], case["ellipticity_deg"], case["mrassumption"]] &) /@ thetaValues;
  listFresnel = {
    curveRows[mats, thetaValues, 0, 2, case["mrassumption"]],
    curveRows[mats, thetaValues, 90, 3, case["mrassumption"]],
    curveRows[mats, thetaValues, 0, 4, case["mrassumption"]],
    curveRows[mats, thetaValues, 90, 5, case["mrassumption"]]
  };

  symbolicAtoms = DeleteDuplicates[Cases[{directRows, listFresnel}, s_Symbol /; Context[s] == "Global`", Infinity]];
  If[Length[symbolicAtoms] > 0,
    Return[
      <|
        "id" -> case["id"],
        "error" -> "Fresnel export contains unresolved Global symbols",
        "symbol_count" -> Length[symbolicAtoms],
        "symbols" -> ToString[symbolicAtoms, InputForm],
        "directRowsInputForm" -> ToString[directRows, InputForm],
        "listFresnelInputForm" -> ToString[listFresnel, InputForm]
      |>
    ]
  ];

  <|
    "id" -> case["id"],
    "inputs" -> <|
      "theta_deg" -> thetaValues,
      "phi_deg" -> case["phi_deg"],
      "ellipticity_deg" -> case["ellipticity_deg"],
      "mrassumption" -> case["mrassumption"]
    |>,
    "batch_export_notes" -> <|
      "gui_formula" -> "GUI listFresnel uses columns {Rp, Rs, Tp, Ts} from FresnelList with phi substitutions {0,90,0,90} degree and ellipticity set to 0.",
      "batch_workaround" -> "This exporter evaluates those four GUI substitutions numerically before each Fresnel call to avoid carrying symbolic phi through the multilayer solver.",
      "empty_wave_guard" -> "safeFresnelRow mirrors SHAARP.ml Fresnel but treats non-Association wave slots as zero contribution before summing A fields."
    |>,
    "mathematica_outputs" -> <|
      "FresnelListDirect" -> jsonValue[directRows],
      "listFresnel_labels" -> {"Rp", "Rs", "Tp", "Ts"},
      "listFresnel" -> jsonValue[listFresnel]
    |>
  |>
];

DeleteFile[diagnosticPath] /; FileExistsQ[diagnosticPath];
PutAppend["Starting Fresnel sweep export with " <> ToString[Length[inputPayload["cases"]]] <> " cases", diagnosticPath];
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
    SHAARPPaths`Ref["fresnel_sweep_reference_errors.json"],
    <|
      "source" -> "Mathematica SHAARP.ml Fresnel sweep export diagnostics",
      "status" -> "mathematica_reference_export_failed",
      "wolfram_version" -> $Version,
      "errors" -> errorCases
    |>,
    "JSON"
  ];
  PutAppend["Aborting Fresnel sweep export because unresolved symbols were found.", diagnosticPath];
  Quit[1]
];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb GUI Fresnel listFresnel reference values",
    "status" -> "mathematica_reference_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "input_manifest" -> inputPath,
    "input_manifest_status" -> inputPayload["status"],
    "functionDownValues" -> <|"Fresnel" -> Length[DownValues[Fresnel]], "safeFresnelRow" -> Length[DownValues[safeFresnelRow]], "f4L" -> Length[DownValues[f4L]], "extWave" -> Length[DownValues[extWave]]|>,
    "batch_export_notes" -> {
      "Raw Mathematica Fresnel can leave unresolved A expressions when one returned wave slot is a non-Association empty value.",
      "safeFresnelRow uses SHAARP.ml f4L values but contributes zero for non-Association wave slots before summing reflected/transmitted A fields.",
      "The GUI-shaped listFresnel curves are evaluated numerically at phi={0,90,0,90} degrees and ellipticity 0."
    },
    "case_count" -> Length[cases],
    "suites" -> {
      <|
        "suite_id" -> "fresnel_sweep_gui_workflow",
        "items" -> cases
      |>
    }
  |>,
  "JSON"
];
Quit[]
