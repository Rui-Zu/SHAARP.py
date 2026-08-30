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

(* Drive the existing Maker-fringe exporter against the ml3 TWO-ACTIVE-LAYER
   manifest (air / active-film-1 / active-film-2 / substrate) by setting
   SHAARPPathOverrides, then Get the base exporter. The base exporter's
   makeMaterial /@ case["layers"] sets pg[[-1]]=nonlinear_flag per layer and f4NL
   builds nonlinear sources for EVERY active layer, so two active films are
   handled with no exporter changes. All paths are ORIGINAL-tree. *)
SHAARPPathOverrides`MakerInputPath = SHAARPPaths`Ref["maker_mathematica_inputs_ml3.json"];
SHAARPPathOverrides`MakerOutputPath = SHAARPPaths`Ref["maker_fringes_reference_ml3.json"];
SHAARPPathOverrides`MakerDiagnosticPath = SHAARPPaths`Ref["maker_fringes_reference_ml3_diagnostics.txt"];
Get[SHAARPPaths`Ref["export_maker_fringes_reference.wl"]];
