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

(* Validate SHAARP.si's OWN GUI dnewExp rotation loop (not extMater's
   TensorContract form) by extracting the live For-loop box from SHAARP_V1.03.nb
   and running it with an exact-rational crystal->lab matrix `a` and a SYMBOLIC
   crystal SHG tensor `doldExp` (d11..d36). The loop is:

     dnewExp[[ii, T[[jj,kk]]]] += a[[ii,ll]] a[[jj,mm]] a[[kk,nn]]
                                   * doldExp[[ll, T[[mm,nn]]]]
   with T = {{1,6,5},{6,2,4},{5,4,3}} and kk starting at jj.

   This is byte-identical to Python's rotate_d_voigt_symbolic; here we run the
   ACTUAL extracted SI box, not a transcription. *)

nbPath = SHAARPPaths`SI["SHAARP_V1.03/SHAARP_V1.03.nb"];
outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_dnewexp_loop_reference_v1.json"];

nb = Import[nbPath, "NB"];
cells = Cases[nb, Cell[box_, "Input", ___] :> box, Infinity];
boxes = cells[[4]];

(* Extract the outermost For-loop box that builds dnewExp from doldExp via `a`. *)
forBoxes = Cases[
   boxes,
   b : RowBox[{"For", ___}] /; (
       StringContainsQ[ToString[b, InputForm], "dnewExp"] &&
       StringContainsQ[ToString[b, InputForm], "doldExp"] &&
       StringContainsQ[ToString[b, InputForm], "\"a\""]
     ) :> b,
   Infinity
];
(* The outermost (ii) loop is the longest such box. *)
loopBox = First[SortBy[forBoxes, -StringLength[ToString[#, InputForm]] &]];

(* Inputs: same exact-rational QC as the validated d-rotation case A. *)
rzA = {{3/5, -4/5, 0}, {4/5, 3/5, 0}, {0, 0, 1}};
rxA = {{1, 0, 0}, {0, 5/13, -12/13}, {0, 12/13, 5/13}};
a = rzA . rxA;
T = {{1, 6, 5}, {6, 2, 4}, {5, 4, 3}};
doldExp = Array[Symbol["d" <> ToString[#1] <> ToString[#2]] &, {3, 6}];
dnewExp = ConstantArray[0, {3, 6}];

ReleaseHold[ToExpression[loopBox, StandardForm, HoldComplete]];  (* mutates dnewExp *)

jsonExpr[expr_] := ToString[InputForm[expr]];
flat = Flatten[dnewExp];
symbolRoles = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> ("Subscript[d," <> ToString[i] <> ToString[j] <> "]"), {i, 1, 3}, {j, 1, 6}]
];
numericSub1 = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> 0.07 i - 0.03 j, "imag" -> 0.011 i j|>, {i, 1, 3}, {j, 1, 6}]
];
numericSub2 = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> -0.02 i + 0.05 j, "imag" -> -0.013 (i + j)|>, {i, 1, 3}, {j, 1, 6}]
];

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.si SHAARP_V1.03.nb live-extracted dnewExp rotation For-loop (run with rational a, symbolic doldExp)",
  "status" -> "mathematica_symbolic_reference_exported",
  "wolfram_version" -> $Version,
  "notebook_path" -> nbPath,
  "for_box_count" -> Length[forBoxes],
  "a_inputform" -> jsonExpr[a],
  "a_orthonormal" -> (a . Transpose[a] == IdentityMatrix[3]),
  "expressions" -> {
    <|
      "expression_id" -> "dnewExp_loop_rotated_d_voigt",
      "source_stage" -> "SHAARP_si_dnewExp_gui_for_loop",
      "output_component" -> "lab_frame_voigt_d_3x6_row_major_flat",
      "expression_format" -> "mathematica_input_form",
      "expression" -> (jsonExpr /@ flat),
      "symbol_roles" -> symbolRoles,
      "numeric_substitutions" -> {numericSub1, numericSub2}
    |>
  }
|>;

exportResult = Export[outputPath, payload, "JSON"];
Print["for_box_count=", Length[forBoxes], " export=", exportResult];
Quit[]
