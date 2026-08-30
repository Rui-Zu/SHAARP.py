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

outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_point_group_d_probe_v1.json"];

dSymbols = Array[Symbol["d" <> ToString[#1] <> ToString[#2]] &, {3, 6}];
eps = IdentityMatrix[3];
lat = {1., 1., 1., 90., 90., 90.};
qc = IdentityMatrix[3];
orientation = {2, {{0, 0, 1}, {0, 1, 0}}, IdentityMatrix[3]};

jsonExpr[expr_] := ToString[InputForm[expr]];
makeCase[pg_] := Module[{mat},
  mat = setMater[{
      "symbolic-" <> pg,
      orientation,
      {pg, dSymbols, 1},
      lat,
      eps,
      eps,
      dSymbols,
      0.,
      qc
    }];
  extMater[mat];
  <|
    "point_group" -> pg,
    "dC" -> (jsonExpr /@ mat[dC]),
    "dL" -> (jsonExpr /@ mat[dL])
  |>
];

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.ml setup.nb symbolic point-group d probe",
  "status" -> "diagnostic_only_ml_material_extension_does_not_apply_gui_point_group_patterns",
  "note" -> "This probe checks whether SHAARP.ml setMater/extMater impose point-group SHG tensor patterns. They do not: dC is supplied by the material record and is not rewritten from pg here. GUI/partial-analytical point-group tensors must be validated from the GUI expression branch instead.",
  "wolfram_version" -> $Version,
  "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
  "functionDownValues" -> <|"setMater" -> Length[DownValues[setMater]], "extMater" -> Length[DownValues[extMater]]|>,
  "cases" -> (makeCase /@ {"1", "3m", "32", "6mm", "\!\(\*OverscriptBox[\(4\), \(_\)]\)3m"})
|>;

exportResult = Export[outputPath, payload, "JSON"];
If[exportResult === $Failed, Quit[2]];
Quit[]
