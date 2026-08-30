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

contractPath = SHAARPPaths`Ref["shaarp_si_stage_export_contract_v1.json"];
outputPath = SHAARPPaths`Ref["shaarp_si_stage_reference_values_v1.json"];
contract = Import[contractPath, "RawJSON"];

(*
  Template only. This file intentionally does not claim validation.
  Next implementation step: load/evaluate the verified SHAARP.si active numeric route,
  assign each requested case input, then replace the missing-value placeholders below
  with actual Mathematica values obtained from the original SHAARP.si equations.
*)

exportableSymbols = {"Croots","E2wroots","ER2w","ERpNum","ERsNum","ETweo","ETwo","EwrootsNum","Fresnel","HR2w","P2eo","P2o","Peoo","RSignalCrystal","ThetaExt","ThetaOrd","Vktweo","Vktwo","nwNumExt","nwNumOrd"};
unresolvedSymbols = {"P2e","Peo","I2"};

Export[
  outputPath,
  <|
    "source" -> "SHAARP.si stage reference export template",
    "status" -> "mathematica_reference_export_template_not_values",
    "validation_claim" -> False,
    "wolfram_version" -> $Version,
    "contract_path" -> contractPath,
    "exportable_symbols" -> exportableSymbols,
    "unresolved_symbols" -> unresolvedSymbols,
    "case_count" -> contract["case_count"],
    "cases" -> {}
  |>,
  "JSON"
];
Quit[]
