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
outputPath = SHAARPPaths`Ref["maker_high_oblique_12_solvefresneln_system_v1.json"];
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

equationLabels[nM_Integer] := Flatten[
  Join[
    {("top:" <> #) & /@ {"E_x", "E_y", "H_x", "H_y"}},
    Table[
      ("layer_" <> ToString[i] <> "_to_layer_" <> ToString[i + 1] <> ":" <> #) & /@ {"E_x", "E_y", "H_x", "H_y"},
      {i, 1, nM - 1}
    ],
    {("bottom:" <> #) & /@ {"E_x", "E_y", "H_x", "H_y"}}
  ]
];

solveFresnelNSystemSnapshot[wAir_, wMed_, wSub_, thickness_, unknowns_] := Module[
  {equ = {}, equTop, equBot, equE, equH, nM, i, getH, B, M, sol, coeffs,
   residualPlus, residualMinus, singVals, basisRecord, basisRecords},
  nM = Length[wMed];
  getH[wav_] := Cross[wav[k], wav[EE]]/(wav[\[Omega]] \[Mu]0);
  basisRecord[label_, wav_, unk_] := <|
    "label" -> label,
    "unknown_symbol" -> ToString[unk, InputForm],
    "omega" -> jsonValue[wav[\[Omega]]],
    "theta" -> jsonValue[wav[\[Theta]]],
    "k" -> jsonValue[wav[k]],
    "k0" -> jsonValue[wav[k0]],
    "electric_basis" -> jsonValue[N[Coefficient[wav[EE], unk], 17]],
    "H_basis" -> jsonValue[N[Coefficient[getH[wav], unk], 17]]
  |>;

  equTop = {
    (Total[Key[EE] /@ wAir][[1 ;; 2]]) == ((Total[Key[EE] /@ wMed[[1]]])[[1 ;; 2]]),
    (Total[getH /@ wAir][[1 ;; 2]]) == ((Total[getH /@ wMed[[1]]])[[1 ;; 2]])
  };
  AppendTo[equ, equTop];

  For[i = 1, i < nM, i++,
    equE = ((Total[(Exp[I (#[k][[3]] thickness[[i]])] #[EE]) & /@ wMed[[i]]])[[1 ;; 2]]) ==
      ((Total[Key[EE] /@ wMed[[i + 1]]])[[1 ;; 2]]);
    equH = ((Total[(Exp[I (#[k][[3]] thickness[[i]])] getH[#]) & /@ wMed[[i]]])[[1 ;; 2]]) ==
      ((Total[getH /@ wMed[[i + 1]]])[[1 ;; 2]]);
    AppendTo[equ, {equE, equH}];
  ];

  equBot = {
    ((Total[(Exp[I (#[k][[3]] thickness[[nM]])] #[EE]) & /@ wMed[[nM]]])[[1 ;; 2]]) ==
      (Total[Key[EE] /@ wSub][[1 ;; 2]]),
    ((Total[(Exp[I (#[k][[3]] thickness[[nM]])] getH[#]) & /@ wMed[[nM]]])[[1 ;; 2]]) ==
      (Total[getH /@ wSub][[1 ;; 2]])
  };
  AppendTo[equ, equBot];

  equ = Flatten[equ];
  equ = Flatten@(Thread /@ equ);
  {B, M} = Normal @ CoefficientArrays[equ, unknowns];
  sol = Quiet @ NSolve[equ, unknowns];
  coeffs = If[Length[sol] > 0, N[unknowns /. sol[[1]], 17], ConstantArray[Indeterminate, Length[unknowns]]];
  residualPlus = If[VectorQ[coeffs, NumericQ], N[M . coeffs + B, 17], {}];
  residualMinus = If[VectorQ[coeffs, NumericQ], N[M . coeffs - B, 17], {}];
  singVals = Quiet @ Check[SingularValueList[N[M, 17]], {}];
  basisRecords = Join[
    MapIndexed[basisRecord["top_unknown_" <> ToString[#2[[1]]], #1, unknowns[[#2[[1]]]]] &, Rest[wAir]],
    Flatten[
      Table[
        MapIndexed[
          basisRecord[
            "layer_" <> ToString[layerIndex] <> "_unknown_" <> ToString[#2[[1]]],
            #1,
            unknowns[[Length[Rest[wAir]] + 4 (layerIndex - 1) + #2[[1]]]]
          ] &,
          wMed[[layerIndex]]
        ],
        {layerIndex, 1, nM}
      ],
      1
    ],
    MapIndexed[
      basisRecord[
        "substrate_unknown_" <> ToString[#2[[1]]],
        #1,
        unknowns[[Length[Rest[wAir]] + 4 nM + #2[[1]]]]
      ] &,
      wSub
    ]
  ];
  <|
    "status" -> "captured_solvefresneln_system_before_original_solve",
    "n_layers" -> nM,
    "unknown_symbols" -> (ToString[#, InputForm] & /@ unknowns),
    "equation_labels" -> equationLabels[nM],
    "coefficient_arrays_B" -> jsonValue[N[B, 17]],
    "coefficient_arrays_M" -> jsonValue[N[M, 17]],
    "nsolve_solution_count" -> Length[sol],
    "nsolve_coefficients" -> jsonValue[coeffs],
    "residual_M_dot_coeffs_plus_B" -> jsonValue[residualPlus],
    "residual_M_dot_coeffs_minus_B" -> jsonValue[residualMinus],
    "singular_values" -> jsonValue[N[singVals, 17]],
    "unknown_wave_basis_records" -> basisRecords
  |>
];

DownValues[originalSolveFresnelNForExport] = DownValues[solveFresnelN] /. solveFresnelN -> originalSolveFresnelNForExport;
Attributes[originalSolveFresnelNForExport] = Attributes[solveFresnelN];
ClearAll[solveFresnelN];
SetAttributes[solveFresnelN, HoldAll];
solveFresnelN[wAir_, wMed_, wSub_, thickness_, unknowns_] := (
  SHAARPReferenceLoader`capturedSolveFresnelNSystem =
    solveFresnelNSystemSnapshot[wAir, wMed, wSub, thickness, unknowns];
  originalSolveFresnelNForExport[wAir, wMed, wSub, thickness, unknowns]
);

case = SelectFirst[inputPayload["cases"], #["id"] == "maker_fringes_moderate_high_oblique" &];
If[MissingQ[case],
  Print["Missing maker_fringes_moderate_high_oblique in input manifest."];
  Quit[1]
];

mats = makeMaterial /@ case["layers"];
\[Omega]0 = N[case["omega0"], 17];
\[CapitalDelta]\[Delta] = N[case["ellipticity_deg"] Degree, 17];
\[CurlyPhi] = N[case["phi_deg"] Degree, 17];
\[Psi] = N[case["psi_deg"] Degree, 17];
n0 = Sqrt[Mean[Eigenvalues[mats[[1]][\[CurlyEpsilon]\[Omega]C]]]];
winhAssumption = case["winhAssumption"];
flagBackward = False;
flagStandingWave = False;
flagJK = False;
flagHH = False;

thetaDeg = 12.;
wIncLocal = setwInc[\[Omega]0, N[(thetaDeg/180) Pi, 17], \[CapitalDelta]\[Delta], \[CurlyPhi], n0];
extWave @ wIncLocal;
wavLLocal = f4L[mats, wIncLocal];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveFresnelN system captured inside f4L for Maker high-oblique theta=12",
    "status" -> "mathematica_maker_high_oblique_12_solvefresneln_system_exported",
    "wolfram_version" -> $Version,
    "input_manifest" -> inputPath,
    "case_id" -> "maker_fringes_moderate_high_oblique",
    "source_case_id" -> "maker_fringes_moderate_high_oblique",
    "theta_deg" -> thetaDeg,
    "solvefresneln_system" -> SHAARPReferenceLoader`capturedSolveFresnelNSystem
  |>,
  "JSON"
];
Quit[]
