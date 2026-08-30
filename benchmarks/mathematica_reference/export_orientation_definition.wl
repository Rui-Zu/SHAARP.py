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

sym[codes_List] := ToExpression["Global`" <> FromCharacterCode[codes, "Unicode"]];

orientationKey = sym[{111, 114, 105, 101, 110, 116, 97, 116, 105, 111, 110}];
pgKey = sym[{112, 103}];
latconKey = sym[{108, 97, 116, 99, 111, 110}];
qcKey = sym[{81, 67}];
qpOmegaKey = sym[{81, 80, 969}];
qp2OmegaKey = sym[{81, 80, 50, 969}];
epsOmegaCKey = sym[{949, 969, 67}];
eps2OmegaCKey = sym[{949, 50, 969, 67}];

Export[
  SHAARPPaths`Ref["orientation_definition.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb orientation definition inspection",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "downValueCounts" -> <|
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
      "epsilon2OmegaC" -> ToCharacterCode[SymbolName[eps2OmegaCKey], "Unicode"]
    |>,
    "definitions" -> <|
      "hklConvert" -> ToString[InputForm[DownValues[hklConvert]]],
      "QC2QP" -> ToString[InputForm[DownValues[QC2QP]]],
      "extMater" -> ToString[InputForm[DownValues[extMater]]]
    |>
  |>,
  "JSON"
];
Quit[]
