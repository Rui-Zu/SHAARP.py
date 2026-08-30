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

(* Extract SHAARP.si's point-group SHG Voigt templates (doldExp) LIVE from the
   SHAARP_V1.03.nb GUI cell. ToExpression of the whole 1 MB cell fails (it holds
   Manipulate/Dynamic boxes), but Cases on the BOX TREE always works: we pull out
   the small, GUI-free `If[PointGroup[[1]] == "PG", doldExp = M]` sub-boxes and
   ToExpression only those. Subscript[d, ij] -> symbol "dij" to match Python. *)

nbPath = SHAARPPaths`SI["SHAARP_V1.03/SHAARP_V1.03.nb"];
outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_doldexp_reference_v1.json"];

nb = Import[nbPath, "NB"];
cells = Cases[nb, Cell[box_, "Input", ___] :> box, Infinity];
boxes = cells[[4]];

(* Minimal If-boxes that assign doldExp (no nested GUI; small). *)
ifBoxes = Cases[
   boxes,
   b : RowBox[{"If", ___}] /; (
       StringContainsQ[ToString[b, InputForm], "doldExp"] &&
       StringContainsQ[ToString[b, InputForm], "PointGroup"] &&
       StringLength[ToString[b, InputForm]] < 6000
     ) :> b,
   Infinity
];

jsonExpr[expr_] := ToString[InputForm[expr]];
symbolRoles = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> ("Subscript[d," <> ToString[i] <> ToString[j] <> "]"),
     {i, 1, 3}, {j, 1, 6}]
];
numericSub1 = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> 0.07 i - 0.03 j, "imag" -> 0.011 i j|>, {i, 1, 3}, {j, 1, 6}]
];
numericSub2 = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> -0.02 i + 0.05 j, "imag" -> -0.013 (i + j)|>, {i, 1, 3}, {j, 1, 6}]
];

extractRecord[ifBox_] := Module[{heldIf, condStrings, pg, rhs, mat, flat},
  heldIf = ToExpression[ifBox, StandardForm, HoldComplete];
  If[heldIf === $Failed, Return[Null]];
  (* heldIf = HoldComplete[If[cond, doldExp = rhs, ...]] *)
  rhs = Cases[heldIf, HoldPattern[Set[doldExp, r_]] :> r, Infinity];
  If[Length[rhs] == 0, Return[Null]];
  condStrings = Cases[heldIf, HoldPattern[If[c_, __]] :> Cases[Hold[c], _String, Infinity], Infinity];
  pg = If[Length[condStrings] >= 1 && Length[condStrings[[1]]] >= 1, Last[condStrings[[1]]], "UNKNOWN"];
  mat = rhs[[1]] /. Subscript[d, n_] :> Symbol["d" <> ToString[n]];
  If[! MatchQ[Dimensions[mat], {3, 6}], Return[Null]];
  flat = Flatten[mat];
  <|
    "expression_id" -> ("doldExp_" <> pg),
    "point_group" -> pg,
    "source_stage" -> "SHAARP_si_doldExp_point_group_template",
    "output_component" -> "crystal_frame_voigt_d_3x6_row_major_flat",
    "expression_format" -> "mathematica_input_form",
    "expression" -> (jsonExpr /@ flat),
    "symbol_roles" -> symbolRoles,
    "numeric_substitutions" -> {numericSub1, numericSub2}
  |>
];

records = DeleteCases[extractRecord /@ ifBoxes, Null];
(* De-duplicate by point group (keep first occurrence). *)
records = DeleteDuplicatesBy[records, #["point_group"] &];
records = Select[records, #["point_group"] =!= "UNKNOWN" &];

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.si SHAARP_V1.03.nb live-extracted doldExp point-group SHG Voigt templates",
  "status" -> "mathematica_symbolic_reference_exported",
  "wolfram_version" -> $Version,
  "notebook_path" -> nbPath,
  "if_box_count" -> Length[ifBoxes],
  "point_groups" -> (#["point_group"] & /@ records),
  "record_count" -> Length[records],
  "expressions" -> records
|>;

exportResult = Export[outputPath, payload, "JSON"];
Print["ifBoxes = ", Length[ifBoxes], " records = ", Length[records], " export = ", exportResult];
Quit[]
