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

\[Mu]0 = 1;
flagLinearSolve = True;
flagNSolve = True;
debug = False;

pythonBenchmarkPath = SHAARPPaths`Repo["benchmarks/multilayer_boundary_system_benchmarks_v1.json"];
outputPath = SHAARPPaths`Ref["solve_fresnel_n_reference_v1.json"];
pythonPayload = Import[pythonBenchmarkPath, "RawJSON"];

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
fromJsonComplex[x_Association] := N[x["real"] + I*x["imag"], 17];

wave[n_, theta_, pol_, amp_, zsign_, omega_] := Module[{dir, e},
  dir = {Sin[theta], 0, zsign*Cos[theta]};
  e = If[pol == "s", {0, 1, 0}, {zsign*Cos[theta], 0, -Sin[theta]}];
  <|\[Omega] -> omega, k -> n*omega*dir, EE -> amp*e|>
];

setElectric[wav_, electric_] := ReplacePart[wav, Key[EE] -> electric];
scaleElectric[wav_, factor_] := setElectric[wav, factor*Key[EE][wav]];

snellTheta[n0_, n1_, theta0_] := ArcSin[n0*Sin[theta0]/n1];
getH[wav_] := Cross[Key[k][wav], Key[EE][wav]]/(Key[\[Omega]][wav]*\[Mu]0);
tangential[waves_] := Join[Total[Key[EE] /@ waves][[1 ;; 2]], Total[getH /@ waves][[1 ;; 2]]];
propagatedTangential[waves_, thickness_] := Module[{phased},
  phased = (scaleElectric[#, Exp[I*(Key[k][#] . {0, 0, 1})*thickness]] &) /@ waves;
  tangential[phased]
];

boundaryResidual[wAir_, wMed_, wSub_, thicknesses_] := Module[{residuals, i, nM},
  nM = Length[wMed];
  residuals = {tangential[wAir] - tangential[wMed[[1]]]};
  For[i = 1, i < nM, i++,
    AppendTo[residuals, propagatedTangential[wMed[[i]], thicknesses[[i]]] - tangential[wMed[[i + 1]]]]
  ];
  AppendTo[residuals, propagatedTangential[wMed[[nM]], thicknesses[[nM]]] - tangential[wSub]];
  Flatten[residuals]
];

solveFresnelNSystem[wAir_, wMed_, wSub_, thicknesses_, unknowns_] := Module[
  {residual, equations, B, M, sol, coefficients, residualAfterSolve},
  residual = boundaryResidual[wAir, wMed, wSub, thicknesses];
  equations = Thread[residual == ConstantArray[0, Length[residual]]];
  {B, M} = Normal[CoefficientArrays[equations, unknowns]];
  coefficients = LinearSolve[M, -B];
  sol = Thread[unknowns -> coefficients];
  residualAfterSolve = residual /. sol;
  <|
    "matrix" -> M,
    "rhs" -> -B,
    "knownResidual" -> B,
    "coefficients" -> coefficients,
    "residual" -> residualAfterSolve,
    "solutionRules" -> sol
  |>
];

complexWaveJson[wav_] := <|
  "omega" -> jsonValue[Key[\[Omega]][wav]],
  "k" -> jsonValue[Key[k][wav]],
  "electric" -> jsonValue[Key[EE][wav]],
  "magnetic" -> jsonValue[getH[wav]],
  "tangential_eh" -> jsonValue[tangential[{wav}]]
|>;

equationMetadata[nLayers_] := Module[{interfaces, rows = {}, components, iface, comp},
  interfaces = Join[{"top"}, Table["layer_" <> ToString[i] <> "_to_layer_" <> ToString[i + 1], {i, 1, nLayers - 1}], {"bottom"}];
  components = {"E_x", "E_y", "H_x", "H_y"};
  Do[
    AppendTo[
      rows,
      <|
        "row_index" -> Length[rows],
        "label" -> iface <> ":" <> comp,
        "interface" -> iface,
        "interface_index" -> interfaceIndex - 1,
        "component" -> comp,
        "field" -> StringTake[comp, 1],
        "axis" -> StringTake[comp, -1]
      |>
    ],
    {interfaceIndex, 1, Length[interfaces]},
    {iface, {interfaces[[interfaceIndex]]}},
    {comp, components}
  ];
  rows
];

layerRole[basisIndex_] := If[basisIndex <= 2, {"forward", basisIndex}, {"backward", basisIndex - 2}];

unknownMetadata[nLayers_] := Module[{rows = {}, role, modeIndex, label},
  Do[
    label = "top_reflected_mode_" <> ToString[modeIndex];
    AppendTo[rows, <|"column_index" -> Length[rows], "label" -> label, "region" -> "top", "layer_index" -> Null, "basis_index" -> modeIndex, "propagation" -> "reflected", "mode_index" -> modeIndex|>],
    {modeIndex, 1, 2}
  ];
  Do[
    role = layerRole[basisIndex];
    label = "layer_" <> ToString[layerIndex] <> "_" <> role[[1]] <> "_mode_" <> ToString[role[[2]]];
    AppendTo[rows, <|"column_index" -> Length[rows], "label" -> label, "region" -> "layer", "layer_index" -> layerIndex, "basis_index" -> basisIndex, "propagation" -> role[[1]], "mode_index" -> role[[2]]|>],
    {layerIndex, 1, nLayers},
    {basisIndex, 1, 4}
  ];
  Do[
    label = "substrate_forward_mode_" <> ToString[modeIndex];
    AppendTo[rows, <|"column_index" -> Length[rows], "label" -> label, "region" -> "substrate", "layer_index" -> Null, "basis_index" -> modeIndex, "propagation" -> "forward", "mode_index" -> modeIndex|>],
    {modeIndex, 1, 2}
  ];
  rows
];

knownSources[layerBasis_, enabled_] := Module[{sources = {}, base, electric},
  If[!TrueQ[enabled], Return[Table[{}, {Length[layerBasis]}]]];
  Do[
    base = layerBasis[[layerIndex, Mod[layerIndex - 1, Length[layerBasis[[layerIndex]]]] + 1]];
    electric = {
      0.02*layerIndex + 0.015*I,
      -0.01*I*(layerIndex + 1),
      0.008 - 0.006*I*layerIndex
    };
    AppendTo[sources, {setElectric[base, electric]}],
    {layerIndex, 1, Length[layerBasis]}
  ];
  sources
];

unknownWaveRecords[metadata_, topBasis_, layerBasis_, substrateBasis_] := Module[{waves},
  waves = Join[topBasis, Flatten[layerBasis, 1], substrateBasis];
  MapThread[<|"metadata" -> #1, "wave" -> complexWaveJson[#2]|> &, {metadata, waves}]
];

knownWaveRecords[topKnown_, layerKnown_] := Module[{records = {}, layerIndex, basisIndex},
  Do[
    AppendTo[records, <|"region" -> "top", "layer_index" -> Null, "basis_index" -> basisIndex, "role" -> "known_top_wave", "wave" -> complexWaveJson[topKnown[[basisIndex]]]|>],
    {basisIndex, 1, Length[topKnown]}
  ];
  Do[
    Do[
      AppendTo[records, <|"region" -> "layer", "layer_index" -> layerIndex, "basis_index" -> basisIndex, "role" -> "known_layer_source", "wave" -> complexWaveJson[layerKnown[[layerIndex, basisIndex]]]|>],
      {basisIndex, 1, Length[layerKnown[[layerIndex]]]}
    ],
    {layerIndex, 1, Length[layerKnown]}
  ];
  records
];

makeCase[case_Association] := Module[
  {
    inputs, caseId, theta, omega, topIndex, layerIndices, substrateIndex, thicknesses,
    layerThetas, substrateTheta, topKnown, topBasis, layerBasis, substrateBasis,
    layerKnown, unknowns, topSymbolic, layerSymbolic, substrateSymbolic,
    metadata, equationRows, system, wAirCall, wMedCall, wSubCall
  },
  inputs = case["inputs"];
  caseId = case["id"];
  theta = N[case["theta_deg"]*Degree, 17];
  omega = fromJsonComplex[inputs["omega"]];
  topIndex = fromJsonComplex[inputs["top_index"]];
  layerIndices = fromJsonComplex /@ inputs["layer_indices"];
  substrateIndex = fromJsonComplex[inputs["substrate_index"]];
  thicknesses = N[inputs["thicknesses"], 17];
  layerThetas = (snellTheta[topIndex, #, theta] &) /@ layerIndices;
  substrateTheta = snellTheta[topIndex, substrateIndex, theta];

  topKnown = {wave[topIndex, theta, "p", 1, 1, omega]};
  topBasis = {
    wave[topIndex, theta, "s", 1, -1, omega],
    wave[topIndex, theta, "p", 1, -1, omega]
  };
  layerBasis = MapThread[
    {
      wave[#1, #2, "s", 1, 1, omega],
      wave[#1, #2, "p", 1, 1, omega],
      wave[#1, #2, "s", 1, -1, omega],
      wave[#1, #2, "p", 1, -1, omega]
    } &,
    {layerIndices, layerThetas}
  ];
  substrateBasis = {
    wave[substrateIndex, substrateTheta, "s", 1, 1, omega],
    wave[substrateIndex, substrateTheta, "p", 1, 1, omega]
  };
  layerKnown = knownSources[layerBasis, case["known_source"]];

  unknowns = Array[Unique["u"] &, Length[Join[topBasis, Flatten[layerBasis, 1], substrateBasis]]];
  topSymbolic = MapThread[scaleElectric, {topBasis, unknowns[[1 ;; Length[topBasis]]]}];
  layerSymbolic = Partition[
    MapThread[scaleElectric, {Flatten[layerBasis, 1], unknowns[[Length[topBasis] + 1 ;; Length[topBasis] + Length[Flatten[layerBasis, 1]]]]}],
    4
  ];
  substrateSymbolic = MapThread[scaleElectric, {substrateBasis, unknowns[[-Length[substrateBasis] ;;]]}];

  wAirCall = Join[topKnown, topSymbolic];
  wMedCall = MapThread[Join, {layerSymbolic, layerKnown}];
  wSubCall = substrateSymbolic;
  system = solveFresnelNSystem[wAirCall, wMedCall, wSubCall, thicknesses, unknowns];

  metadata = unknownMetadata[Length[layerBasis]];
  equationRows = equationMetadata[Length[layerBasis]];
  <|
    "id" -> caseId,
    "inputs" -> <|
      "theta_deg" -> case["theta_deg"],
      "layer_count" -> Length[layerBasis],
      "known_source" -> case["known_source"]
    |>,
    "mathematica_outputs" -> <|
      "solveFresnelN_matrix" -> jsonValue[system["matrix"]],
      "solveFresnelN_rhs" -> jsonValue[system["rhs"]],
      "solveFresnelN_known_residual" -> jsonValue[system["knownResidual"]],
      "solveFresnelN_coefficients" -> jsonValue[system["coefficients"]],
      "solveFresnelN_residual" -> jsonValue[system["residual"]],
      "equation_labels" -> (#[["label"]] &) /@ equationRows,
      "equation_metadata" -> equationRows,
      "unknown_labels" -> (#[["label"]] &) /@ metadata,
      "unknown_metadata" -> metadata,
      "unknown_basis_waves" -> unknownWaveRecords[metadata, topBasis, layerBasis, substrateBasis],
      "known_waves" -> knownWaveRecords[topKnown, layerKnown]
    |>
  |>
];

cases = makeCase /@ pythonPayload["cases"];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveFresnelN boundary-system reference values",
    "status" -> "mathematica_reference_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"solveFresnelN" -> Length[DownValues[solveFresnelN]]|>,
    "case_count" -> Length[cases],
    "suites" -> {
      <|
        "suite_id" -> "multilayer_boundary_system",
        "items" -> cases
      |>
    }
  |>,
  "JSON"
];
Quit[]
