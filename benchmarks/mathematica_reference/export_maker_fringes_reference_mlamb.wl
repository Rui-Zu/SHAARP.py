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

(* Drive the existing Maker-fringe exporter against the mlamb MULTILAYER manifest — the
   F58 certification of a NON-AIR INCIDENT MEDIUM (isotropic n(w) = 1.33, n(2w) = 1.34)
   on mlgraze/ml8's exact 9-layer stacks, at ordinary incidence (theta 20/40/60 deg).

   NO exporter edit is needed and none is made: the base exporter already derives the
   ambient index per case from layer 0 of the manifest —
       n0 = Sqrt[Mean[Eigenvalues[mats[[1]][\[CurlyEpsilon]\[Omega]C]]]]
   (export_maker_fringes_reference.wl:119) — and feeds it to setwInc, which builds the
   incident wave as
       wInc = setWave[{w, n0 (w/c0) {Sin[thetaInc], 0, Cos[thetaInc]}, n0 w/c0, thetaInc, Einc}]
   so n0 scales k and k0 and therefore the tangential k_x that every Snell/Fresnel step
   chains off. Every PRIOR manifest set layer 0 to eps = I, making n0 == 1 identically;
   this manifest sets eps = n^2 I, so the ORIGINAL tool computes the non-air ambient.

   Why this set exists: the RELEASED Mathematica .ml GUI hardcodes BOTH half-spaces to air
   (SHAARP.ml.nb:666-716 forces m1 = setMater@Air[]; :425-439 overwrites the exit slot with
   mbot = Air; the layer selector Range[2, materialnumber+1] reaches neither), while the
   SHAARP.py GUI lets the user set them. That extension needs live evidence, and the engine
   itself was always general (setup.nb:6421: wSub = "waves into Substrate (can be Air)").

   All paths are ORIGINAL-tree. *)
SHAARPPathOverrides`MakerInputPath = SHAARPPaths`Ref["maker_mathematica_inputs_mlamb.json"];
SHAARPPathOverrides`MakerOutputPath = SHAARPPaths`Ref["maker_fringes_reference_mlamb.json"];
SHAARPPathOverrides`MakerDiagnosticPath = SHAARPPaths`Ref["maker_fringes_reference_mlamb_diagnostics.txt"];
Get[SHAARPPaths`Ref["export_maker_fringes_reference.wl"]];
