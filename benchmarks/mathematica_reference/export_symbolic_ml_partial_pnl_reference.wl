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

outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_ml_partial_pnl_reference_v1.json"];

dSymbols = Array[Symbol["d" <> ToString[#1] <> ToString[#2]] &, {3, 6}];
field[label_] := {Symbol[label <> "x"], Symbol[label <> "y"], Symbol[label <> "z"]};
fields = <|
  "F1" -> field["F1"],
  "F2" -> field["F2"],
  "B1" -> field["B1"],
  "B2" -> field["B2"]
|>;
fieldLabels = {"F1", "F2", "B1", "B2"};
fieldAxes = {"x", "y", "z"};

sourceSpecs = {
  {"PNLF1F1", "F1F1", "F1", "F1", 1},
  {"PNLF2F2", "F2F2", "F2", "F2", 1},
  {"PNLF1F2", "F1F2", "F1", "F2", 2},
  {"PNLB1B1", "B1B1", "B1", "B1", 1},
  {"PNLB2B2", "B2B2", "B2", "B2", 1},
  {"PNLB1B2", "B1B2", "B1", "B2", 2},
  {"PNLF1B1", "F1B1", "F1", "B1", 2},
  {"PNLF2B2", "F2B2", "F2", "B2", 2},
  {"PNLF1B2", "F1B2", "F1", "B2", 2},
  {"PNLF2B1", "F2B1", "F2", "B1", 2}
};

symbolRoles = Association[
  Flatten[
    {
      Table["d" <> ToString[i] <> ToString[j] -> "Subscript[d," <> ToString[i] <> ToString[j] <> "]", {i, 1, 3}, {j, 1, 6}],
      Table[label <> axis -> "Subscript[Superscript[E,omega]," <> label <> "," <> axis <> "]",
        {label, {"F1", "F2", "B1", "B2"}}, {axis, {"x", "y", "z"}}]
    }
  ]
];

numericSubstitutions = {
  Association[
    Flatten[
      {
        Table["d" <> ToString[i] <> ToString[j] -> <|"real" -> N[0.031 i - 0.017 j, 17], "imag" -> N[0.007 i j, 17]|>, {i, 1, 3}, {j, 1, 6}],
        Table[
          With[{label = fieldLabels[[idx]], axis = fieldAxes[[axisIdx]]},
            label <> axis -> <|"real" -> N[0.09 StringLength[label] + 0.03 ToCharacterCode[axis][[1]]/120 - 0.04 idx, 17],
              "imag" -> N[-0.05 idx + 0.02 axisIdx, 17]|>
          ],
          {idx, 1, Length[fieldLabels]}, {axisIdx, 1, Length[fieldAxes]}]
      }
    ]
  ],
  Association[
    Flatten[
      {
        Table["d" <> ToString[i] <> ToString[j] -> <|"real" -> N[-0.041 i + 0.019 j, 17], "imag" -> N[-0.005 (i + 2 j), 17]|>, {i, 1, 3}, {j, 1, 6}],
        Table[
          With[{label = fieldLabels[[idx]], axis = fieldAxes[[axisIdx]]},
            label <> axis -> <|"real" -> N[-0.07 idx + 0.025 ToCharacterCode[axis][[1]]/130, 17],
              "imag" -> N[0.04 idx - 0.015 StringLength[label] - 0.01 axisIdx, 17]|>
          ],
          {idx, 1, Length[fieldLabels]}, {axisIdx, 1, Length[fieldAxes]}]
      }
    ]
  ]
};

record[spec_] := Module[
  {expr},
  expr = Simplify[computePNL[dSymbols, fields[spec[[3]]], fields[spec[[4]]], spec[[5]]]];
  <|
    "expression_id" -> "ml_partial_" <> spec[[2]],
    "source_stage" -> "SHAARP.ml_partial_computePNL",
    "output_component" -> spec[[1]],
    "expression_format" -> "mathematica_input_form",
    "expression" -> (ToString[InputForm[#]] & /@ expr),
    "symbol_roles" -> symbolRoles,
    "numeric_substitutions" -> numericSubstitutions
  |>
];

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.ml setup.nb symbolic ten-source partial PNL export",
  "status" -> "mathematica_symbolic_reference_exported",
  "wolfram_version" -> $Version,
  "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
  "functionDownValues" -> <|"computePNL" -> Length[DownValues[computePNL]]|>,
  "expressions" -> (record /@ sourceSpecs)
|>;

exportResult = Export[outputPath, payload, "JSON"];
If[exportResult === $Failed, Quit[2]];
Quit[]
