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

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;

makeD[seed_] := Table[
  N[(0.11*seed + 0.07*i - 0.03*j) + I*(0.05*seed - 0.02*i + 0.04*j)],
  {i, 1, 3},
  {j, 1, 6}
];

makeE[seed_, offset_] := Table[
  N[(0.19*seed - 0.13*i + 0.07*offset) + I*(0.11*seed + 0.05*i - 0.03*offset)],
  {i, 1, 3}
];

factors = {1, 2, -0.5 + 0.25*I, 0.75 - 1.2*I, -1.1};
cases = Table[
  Module[
    {
      d = makeD[idx],
      e1 = makeE[idx, 1],
      e2 = makeE[idx + 3, 2],
      factor = factors[[1 + Mod[idx - 1, Length[factors]]]],
      pnl
    },
    pnl = computePNL[d, e1, e2, factor];
    <|
      "id" -> StringTemplate["compute_pnl_``"][IntegerString[idx, 10, 2]],
      "inputs" -> <|
        "d_voigt_lab" -> jsonValue[d],
        "e1" -> jsonValue[e1],
        "e2" -> jsonValue[e2],
        "factor" -> jsonValue[factor]
      |>,
      "outputs" -> <|
        "pnl" -> jsonValue[pnl]
      |>
    |>
  ],
  {idx, 1, 20}
];

Export[
  SHAARPPaths`Ref["compute_pnl_reference_v1.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb computePNL reference values",
    "status" -> "mathematica_reference_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|
      "computePNL" -> Length[DownValues[computePNL]]
    |>,
    "cases" -> cases
  |>,
  "JSON"
];
Quit[]
