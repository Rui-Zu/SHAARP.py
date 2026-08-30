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

outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_rotated_d_reference_v1.json"];

(* Two EXACT-RATIONAL crystal->lab rotations, computed as rational matrix
   products IDENTICALLY to the Python generator (avoid hand-transcription):
   A = Rz(3-4-5) . Rx(5-12-13);  B = Rz(15-8-17) . Ry(21-20-29). *)
rzA = {{3/5, -4/5, 0}, {4/5, 3/5, 0}, {0, 0, 1}};
rxA = {{1, 0, 0}, {0, 5/13, -12/13}, {0, 12/13, 5/13}};
qcA = rzA . rxA;
rzB = {{15/17, -8/17, 0}, {8/17, 15/17, 0}, {0, 0, 1}};
ryB = {{21/29, 0, 20/29}, {0, 1, 0}, {-20/29, 0, 21/29}};
qcB = rzB . ryB;

dSymbols = Array[Symbol["d" <> ToString[#1] <> ToString[#2]] &, {3, 6}];
eps = IdentityMatrix[3];
lat = {1, 1, 1, 90, 90, 90};
jsonExpr[expr_] := ToString[InputForm[expr]];

symbolRoles = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> ("Subscript[d," <> ToString[i] <> ToString[j] <> "]"), {i, 1, 3}, {j, 1, 6}]
];
numericSub1 = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> 0.07 i - 0.03 j, "imag" -> 0.011 i j|>, {i, 1, 3}, {j, 1, 6}]
];
numericSub2 = Association @@ Flatten[
   Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> -0.02 i + 0.05 j, "imag" -> -0.013 (i + j)|>, {i, 1, 3}, {j, 1, 6}]
];

makeRecord[id_, qc_] := Module[{mat, dLmat, flat},
  mat = setMater[{
      "symbolic-rotated-d-" <> id,
      {2, {{0, 0, 1}, {0, 1, 0}}, qc},
      {"1", dSymbols, 1},
      lat,
      eps,
      eps,
      dSymbols,
      0.,
      qc
    }];
  extMater[mat];
  dLmat = mat[dL];
  flat = Flatten[dLmat];
  <|
    "expression_id" -> id,
    "source_stage" -> "extMater_dL_dnewExp",
    "output_component" -> "lab_frame_voigt_d_3x6_row_major_flat",
    "expression_format" -> "mathematica_input_form",
    "expression" -> (jsonExpr /@ flat),
    "symbol_roles" -> symbolRoles,
    "numeric_substitutions" -> {numericSub1, numericSub2},
    "mat_qc_inputform" -> jsonExpr[mat[QC]]
  |>
];

recA = makeRecord["rotated_d_voigt_symbolic", qcA];
recB = makeRecord["rotated_d_voigt_symbolic_b", qcB];

(* verify both rotations are exactly orthonormal before trusting the export *)
orthoA = qcA . Transpose[qcA] == IdentityMatrix[3];
orthoB = qcB . Transpose[qcB] == IdentityMatrix[3];

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.ml setup.nb symbolic crystal-to-lab Voigt-d rotation (extMater dL)",
  "status" -> "mathematica_symbolic_reference_exported",
  "wolfram_version" -> $Version,
  "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
  "functionDownValues" -> <|"setMater" -> Length[DownValues[setMater]], "extMater" -> Length[DownValues[extMater]]|>,
  "qcA_inputform" -> jsonExpr[qcA],
  "qcB_inputform" -> jsonExpr[qcB],
  "qcA_orthonormal" -> orthoA,
  "qcB_orthonormal" -> orthoB,
  "expressions" -> {recA, recB}
|>;

exportResult = Export[outputPath, payload, "JSON"];
Print["exportResult = ", exportResult, " orthoA=", orthoA, " orthoB=", orthoB];
Quit[]
