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

SHAARPReferenceLoader`outputPath = SHAARPPaths`Ref["orientation_reference_v1.json"];
SHAARPReferenceLoader`diagnosticsPath = SHAARPPaths`Ref["orientation_reference_diagnostics.txt"];
If[FileExistsQ[SHAARPReferenceLoader`diagnosticsPath], DeleteFile[SHAARPReferenceLoader`diagnosticsPath]];
log[msg_] := PutAppend[DateString[{"ISODateTime"}] <> " " <> ToString[msg, InputForm], SHAARPReferenceLoader`diagnosticsPath];
If[FileExistsQ[SHAARPReferenceLoader`outputPath], DeleteFile[SHAARPReferenceLoader`outputPath]];
log["starting orientation reference export"];

SetDirectory[SHAARPPaths`MLDir[]];
nb = Import[SHAARPPaths`ML["setup.nb"], "NB"];
SHAARPReferenceLoader`cells = Cases[nb, Cell[box_, "Input", ___] :> box, Infinity];
SHAARPReferenceLoader`inputCellCount = Length[SHAARPReferenceLoader`cells];
SHAARPReferenceLoader`neededCellIndices = {2, 4, 5, 6, 39};
log[<|"inputCellCount" -> SHAARPReferenceLoader`inputCellCount, "neededCellIndices" -> SHAARPReferenceLoader`neededCellIndices|>];
Scan[
  (
    log[<|"releasingCell" -> #|>];
    ReleaseHold[ToExpression[SHAARPReferenceLoader`cells[[#]], StandardForm, HoldComplete]];
    log[<|"releasedCell" -> #|>]
  ) &,
  SHAARPReferenceLoader`neededCellIndices
];
log[<|"loadedInputCellCount" -> SHAARPReferenceLoader`inputCellCount, "hklConvertDownValues" -> Length[DownValues[hklConvert]], "QC2QPDownValues" -> Length[DownValues[QC2QP]], "extMaterDownValues" -> Length[DownValues[extMater]]|>];

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
sym[codes_List] := ToExpression["Global`" <> FromCharacterCode[codes, "Unicode"]];

orientationKey = sym[{111, 114, 105, 101, 110, 116, 97, 116, 105, 111, 110}];
pgKey = sym[{112, 103}];
latconKey = sym[{108, 97, 116, 99, 111, 110}];
qcKey = sym[{81, 67}];
qpOmegaKey = sym[{81, 80, 969}];
qp2OmegaKey = sym[{81, 80, 50, 969}];
epsOmegaCKey = sym[{949, 969, 67}];
eps2OmegaCKey = sym[{949, 50, 969, 67}];
epsOmegaLKey = sym[{949, 969, 76}];
eps2OmegaLKey = sym[{949, 50, 969, 76}];
dCKey = sym[{100, 67}];
dLKey = sym[{100, 76}];

makeEpsilon[seed_, imagScale_] := N[{
  {2.0 + 0.19*seed + I*imagScale*(0.021 + 0.003*seed), 0.04 + 0.01*seed - I*imagScale*(0.014 + 0.002*seed), -0.025 + 0.004*seed + I*imagScale*0.011},
  {0.04 + 0.01*seed - I*imagScale*(0.006 + 0.001*seed), 2.6 + 0.13*seed + I*imagScale*(0.026 + 0.002*seed), 0.033 - 0.003*seed + I*imagScale*0.009},
  {-0.025 + 0.004*seed + I*imagScale*0.007, 0.033 - 0.003*seed - I*imagScale*0.013, 3.2 + 0.17*seed + I*imagScale*(0.031 + 0.001*seed)}
}];

makeD[seed_] := Table[
  N[(0.07*seed + 0.013*i - 0.011*j) + I*(0.021*seed - 0.017*i + 0.009*j)],
  {i, 1, 3},
  {j, 1, 6}
];

makeCase[idx_, pointGroup_, latcon_, plane_, vertical_, seed_, imagScale_] := Check[Module[
  {matCase, epsW, eps2W, dC, initialOrientation, convertedOrientation, qc, qpW, qp2W},
  log[<|"caseStart" -> idx, "pointGroup" -> pointGroup, "plane" -> plane, "vertical" -> vertical|>];
  epsW = makeEpsilon[seed, imagScale];
  eps2W = makeEpsilon[seed + 5, imagScale + 0.35];
  dC = makeD[seed];
  initialOrientation = {1, {plane, vertical}, IdentityMatrix[3]};
  With[
    {ok = orientationKey, pk = pgKey, lk = latconKey, ek = epsOmegaCKey, e2k = eps2OmegaCKey, dk = dCKey},
    matCase = <|ok -> initialOrientation, pk -> {pointGroup}, lk -> latcon, ek -> epsW, e2k -> eps2W, dk -> dC|>;
  ];
  log[<|"caseStage" -> idx, "stage" -> "before_hklConvert"|>];
  hklConvert[matCase];
  log[<|"caseStage" -> idx, "stage" -> "after_hklConvert"|>];
  extMater[matCase];
  log[<|"caseStage" -> idx, "stage" -> "after_extMater"|>];
  convertedOrientation = matCase[orientationKey];
  qc = matCase[qcKey];
  qpW = matCase[qpOmegaKey];
  qp2W = matCase[qp2OmegaKey];
  log[<|"caseStage" -> idx, "stage" -> "after_QP_lookup"|>];
  log[<|"caseDone" -> idx|>];
  <|
    "id" -> StringTemplate["orientation_``"][IntegerString[idx, 10, 2]],
    "inputs" -> <|
      "point_group" -> pointGroup,
      "latcon" -> jsonValue[latcon],
      "plane_hkl" -> jsonValue[plane],
      "vertical_uvw" -> jsonValue[vertical],
      "epsilon_omega_crystal" -> jsonValue[epsW],
      "epsilon_2omega_crystal" -> jsonValue[eps2W],
      "d_voigt_crystal" -> jsonValue[dC]
    |>,
    "outputs" -> <|
      "orientation" -> jsonValue[convertedOrientation],
      "orientation_axes" -> jsonValue[convertedOrientation[[3]]],
      "QC" -> jsonValue[qc],
      "QPOmega" -> jsonValue[qpW],
      "QP2Omega" -> jsonValue[qp2W],
      "epsilon_omega_lab" -> jsonValue[matCase[epsOmegaLKey]],
      "epsilon_2omega_lab" -> jsonValue[matCase[eps2OmegaLKey]],
      "d_voigt_lab" -> jsonValue[matCase[dLKey]]
    |>
  |>
], <|
  "id" -> StringTemplate["orientation_``"][IntegerString[idx, 10, 2]],
  "error" -> "Mathematica case generation failed",
  "inputs" -> <|
    "point_group" -> pointGroup,
    "latcon" -> jsonValue[latcon],
    "plane_hkl" -> jsonValue[plane],
    "vertical_uvw" -> jsonValue[vertical]
  |>
|>];

caseSpecs = {
  {"1", {4.1, 5.2, 6.3, 76.0, 88.0, 103.0}, {1, 2, 3}, {1, 1, -1}, 1, 0.0},
  {"1", {4.1, 5.2, 6.3, 76.0, 88.0, 103.0}, {2, 3, 5}, {1, 1, -1}, 2, 1.0},
  {"2", {4.4, 5.1, 6.2, 90.0, 101.0, 90.0}, {2, 1, 3}, {1, 1, -1}, 3, 0.6},
  {"m", {4.7, 5.4, 6.1, 90.0, 96.0, 90.0}, {3, 2, 1}, {1, -2, 1}, 4, 1.2},
  {"mm2", {3.8, 4.7, 5.9, 90.0, 90.0, 90.0}, {1, 3, 2}, {1, -1, 1}, 5, 0.8},
  {"mmm", {3.6, 4.9, 5.4, 90.0, 90.0, 90.0}, {3, 1, 2}, {1, 1, -2}, 6, 1.1},
  {"4mm", {3.9, 3.9, 5.8, 90.0, 90.0, 90.0}, {2, 3, 1}, {1, -1, 1}, 8, 1.3},
  {"m3m", {4.0, 4.0, 4.0, 90.0, 90.0, 90.0}, {1, 2, 3}, {1, 1, -1}, 10, 1.5},
  {"23", {4.3, 4.3, 4.3, 90.0, 90.0, 90.0}, {2, 3, 4}, {1, 2, -2}, 11, 0.7},
  {"432", {4.5, 4.5, 4.5, 90.0, 90.0, 90.0}, {3, 4, 5}, {1, -2, 1}, 12, 1.6}
};
cases = MapIndexed[makeCase[#2[[1]], #[[1]], #[[2]], #[[3]], #[[4]], #[[5]], #[[6]]] &, caseSpecs];
log[<|"caseCount" -> Length[cases], "errorCount" -> Count[cases, _?(AssociationQ[#] && KeyExistsQ[#, "error"] &)]|>];

exportResult = Export[
  SHAARPReferenceLoader`outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb hklConvert/QC2QP orientation reference values",
    "status" -> "mathematica_reference_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|
      "hklConvert" -> Length[DownValues[hklConvert]],
      "QC2QP" -> Length[DownValues[QC2QP]],
      "extMater" -> Length[DownValues[extMater]]
    |>,
    "selectedKeys" -> <|
      "orientation" -> ToCharacterCode[SymbolName[orientationKey], "Unicode"],
      "pg" -> ToCharacterCode[SymbolName[pgKey], "Unicode"],
      "latcon" -> ToCharacterCode[SymbolName[latconKey], "Unicode"],
      "QC" -> ToCharacterCode[SymbolName[qcKey], "Unicode"],
      "QPOmega" -> ToCharacterCode[SymbolName[qpOmegaKey], "Unicode"],
      "QP2Omega" -> ToCharacterCode[SymbolName[qp2OmegaKey], "Unicode"],
      "epsilonOmegaC" -> ToCharacterCode[SymbolName[epsOmegaCKey], "Unicode"],
      "epsilon2OmegaC" -> ToCharacterCode[SymbolName[eps2OmegaCKey], "Unicode"],
      "dC" -> ToCharacterCode[SymbolName[dCKey], "Unicode"],
      "dL" -> ToCharacterCode[SymbolName[dLKey], "Unicode"]
    |>,
    "case_count" -> Length[cases],
    "cases" -> cases
  |>,
  "JSON"
];
log[<|"exportResult" -> exportResult, "outputExists" -> FileExistsQ[SHAARPReferenceLoader`outputPath], "outputByteCount" -> If[FileExistsQ[SHAARPReferenceLoader`outputPath], FileByteCount[SHAARPReferenceLoader`outputPath], Missing["NoFile"]]|>];
Quit[]
