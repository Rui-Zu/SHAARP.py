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

(* Drive the existing Maker-fringe exporter against the mlgraze MULTILAYER manifest
   (GRAZING INCIDENCE, theta 75..83 deg, on ml8's exact 9-layer stacks: air /
   active-film / anisotropic-interlayer / isotropic-interlayer / isotropic-
   interlayer-2 / isotropic-interlayer-3 / anisotropic-interlayer-2 /
   isotropic-interlayer-4 / substrate) by setting SHAARPPathOverrides, then Get
   the base exporter. Identical to the ml8 wrapper except the tag — the stacks
   differ from ml8 ONLY in incidence angle, so a disagreement is attributable to
   the angle. The grid stops at 83 deg because f4NL itself kills the kernel —
   silently, exit 0, no stderr — above a CASE-DEPENDENT onset angle (mapped
   per-angle 2026-08-15: case 1 onset 88.0, case 2 onset 84.0 [binds the common
   grid], case 3 onset > 87.5, case 4 onset 87.0; localized INSIDE f4NL by a
   breadcrumb-instrumented exporter). 83 deg = the largest common band all four
   cases export. See the mlgraze generator + gated test + residual-risk R1.
   All paths are ORIGINAL-tree. *)
SHAARPPathOverrides`MakerInputPath = SHAARPPaths`Ref["maker_mathematica_inputs_mlgraze.json"];
SHAARPPathOverrides`MakerOutputPath = SHAARPPaths`Ref["maker_fringes_reference_mlgraze.json"];
SHAARPPathOverrides`MakerDiagnosticPath = SHAARPPaths`Ref["maker_fringes_reference_mlgraze_diagnostics.txt"];
Get[SHAARPPaths`Ref["export_maker_fringes_reference.wl"]];
