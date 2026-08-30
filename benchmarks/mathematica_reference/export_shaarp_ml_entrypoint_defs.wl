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
held = ToExpression[cells, StandardForm, HoldComplete];
Scan[ReleaseHold, held];

(* Use ToExpression[name, InputForm, Definition] so Definition (HoldAll)
   wraps the symbol BEFORE it evaluates to its OwnValue/Function body. *)
dumpDef[n_] := Module[{full},
  If[! NameQ["Global`" <> n], Return[<|"defined" -> False|>]];
  full = ToExpression["Global`" <> n, InputForm, Definition];
  <|
    "defined" -> True,
    "definition" -> StringTake[ToString[full, InputForm], UpTo[200000]]
  |>
];

wanted = {"Fresnel", "MF", "SampleRotate", "solveFresnel", "solveFresnelN"};
defs = Association @@ Table[n -> dumpDef[n], {n, wanted}];

Export[
  SHAARPPaths`Ref["shaarp_ml_entrypoint_defs.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb entry-point full Definition[] dumps",
    "wolfram_version" -> $Version,
    "definitions" -> defs
  |>,
  "JSON"
];
Quit[]
