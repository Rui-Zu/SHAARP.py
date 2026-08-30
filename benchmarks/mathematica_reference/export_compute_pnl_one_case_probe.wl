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
loadResult = CheckAbort[Quiet[Check[Scan[ReleaseHold, held], $Failed]], $Aborted];

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
jsonValue[x_] := ToString[x, InputForm];

d = {
  {0.11 + 0.02 I, -0.05, 0.04 I, 0.18, -0.09 I, 0.07},
  {-0.03 I, 0.08, -0.06, 0.05 I, 0.04, -0.02},
  {0.13, -0.10 I, 0.16, -0.08, 0.03 I, 0.09}
};
e1 = {0.31 + 0.07 I, -0.22 + 0.04 I, 0.15 - 0.09 I};
e2 = {-0.18 + 0.05 I, 0.27 - 0.02 I, -0.11 + 0.08 I};
factor = -0.5 + 0.25 I;
pnl = CheckAbort[Quiet[Check[computePNL[d, e1, e2, factor], $Failed]], $Aborted];

Export[
  SHAARPPaths`Ref["compute_pnl_one_case_probe_v1.json"],
  <|
    "source" -> "Live Wolfram Mathematica one-case computePNL probe",
    "status" -> "mathematica_reference_exported",
    "batch_execution" -> "file_export_confirmed",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "loadResult" -> ToString[loadResult, InputForm],
    "functionDownValues" -> <|
      "computePNL" -> Length[DownValues[computePNL]]
    |>,
    "diagnostics" -> <|
      "pnlHead" -> ToString[Head[pnl], InputForm],
      "pnlFailed" -> SameQ[pnl, $Failed],
      "pnlAborted" -> SameQ[pnl, $Aborted]
    |>,
    "cases" -> {
      <|
        "id" -> "live_compute_pnl_one_case_probe_001",
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
    }
  |>,
  "JSON"
];
Quit[]
