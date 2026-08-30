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

SetDirectory[SHAARPPaths`MLDir[]];
nb = Import[SHAARPPaths`ML["setup.nb"], "NB"];
cells = Cases[nb, Cell[box_, "Input", ___] :> box, Infinity];
SHAARPReferenceLoader`inputCellCount = Length[cells];
held = ToExpression[cells, StandardForm, HoldComplete];
Scan[ReleaseHold, held];

outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_compute_pnl_reference_v1.json"];

dSymbols = Array[Symbol["d" <> ToString[#1] <> ToString[#2]] &, {3, 6}];
e1Symbols = {e1x, e1y, e1z};
e2Symbols = {e2x, e2y, e2z};

sameSource = Simplify[computePNL[dSymbols, e1Symbols, e1Symbols]];
mixedSource = Simplify[computePNL[dSymbols, e1Symbols, e2Symbols, 2]];

symbolRoles = Association[
  Flatten[
    {
      Table["d" <> ToString[i] <> ToString[j] -> "Subscript[d," <> ToString[i] <> ToString[j] <> "]", {i, 1, 3}, {j, 1, 6}],
      {"e1x" -> "Subscript[Superscript[E,omega],1,x]",
       "e1y" -> "Subscript[Superscript[E,omega],1,y]",
       "e1z" -> "Subscript[Superscript[E,omega],1,z]",
       "e2x" -> "Subscript[Superscript[E,omega],2,x]",
       "e2y" -> "Subscript[Superscript[E,omega],2,y]",
       "e2z" -> "Subscript[Superscript[E,omega],2,z]"}
    }
  ]
];

numericSubstitutions = {
  Association[
    Flatten[
      {
        Table["d" <> ToString[i] <> ToString[j] -> <|"real" -> N[0.07 i - 0.03 j, 17], "imag" -> N[0.011 i j, 17]|>, {i, 1, 3}, {j, 1, 6}],
        {"e1x" -> <|"real" -> 0.23, "imag" -> -0.17|>,
         "e1y" -> <|"real" -> -0.41, "imag" -> 0.09|>,
         "e1z" -> <|"real" -> 0.13, "imag" -> 0.31|>,
         "e2x" -> <|"real" -> -0.29, "imag" -> 0.19|>,
         "e2y" -> <|"real" -> 0.37, "imag" -> -0.23|>,
         "e2z" -> <|"real" -> -0.11, "imag" -> -0.07|>}
      }
    ]
  ],
  Association[
    Flatten[
      {
        Table["d" <> ToString[i] <> ToString[j] -> <|"real" -> N[-0.02 i + 0.05 j, 17], "imag" -> N[-0.013 (i + j), 17]|>, {i, 1, 3}, {j, 1, 6}],
        {"e1x" -> <|"real" -> 0.08, "imag" -> 0.44|>,
         "e1y" -> <|"real" -> 0.51, "imag" -> -0.12|>,
         "e1z" -> <|"real" -> -0.33, "imag" -> 0.27|>,
         "e2x" -> <|"real" -> 0.19, "imag" -> -0.38|>,
         "e2y" -> <|"real" -> -0.47, "imag" -> -0.16|>,
         "e2z" -> <|"real" -> 0.22, "imag" -> 0.29|>}
      }
    ]
  ]
};

record[id_, component_, expr_] := <|
  "expression_id" -> id,
  "source_stage" -> "computePNL",
  "output_component" -> component,
  "expression_format" -> "mathematica_input_form",
  "expression" -> (ToString[InputForm[#]] & /@ expr),
  "symbol_roles" -> symbolRoles,
  "numeric_substitutions" -> numericSubstitutions
|>;

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.ml setup.nb symbolic computePNL export",
  "status" -> "mathematica_symbolic_reference_exported",
  "wolfram_version" -> $Version,
  "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
  "functionDownValues" -> <|"computePNL" -> Length[DownValues[computePNL]]|>,
  "expressions" -> {
    record["compute_pnl_same_symbolic", "same_wave_vector", sameSource],
    record["compute_pnl_mixed_symbolic", "mixed_wave_vector_factor_2", mixedSource]
  }
|>;

exportResult = Export[outputPath, payload, "JSON"];
If[exportResult === $Failed, Quit[2]];
Quit[]
