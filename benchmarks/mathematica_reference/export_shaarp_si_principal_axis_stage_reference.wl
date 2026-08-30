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

mappingPath = SHAARPPaths`Ref["shaarp_si_case_input_mapping_v1.json"];
snippetPath = SHAARPPaths`Ref["shaarp_si_refined_region_texts_v1.json"];
outputPath = SHAARPPaths`Ref["shaarp_si_principal_axis_stage_reference_v1.json"];

mapping = Import[mappingPath, "RawJSON"];
snippets = Import[snippetPath, "RawJSON"];
principalSnippet = SelectFirst[snippets["records"], #["region_id"] == "principal_axis_and_index_setup" &]["text"];

toComplex[x_Association] /; KeyExistsQ[x, "real"] := N[x["real"] + I*x["imag"], 17];
toComplex[x_List] := toComplex /@ x;
toComplex[x_] := x;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;

maxAbs[x_] := Max[Abs[Flatten[N[x]]]];

caseRecord[case_] := Module[
  {vars, expected, epsWLabExpected, eps2WLabExpected, messages = {}, held},
  Clear[
    Inv, aCryToPrinw, aCryToPrin2w, Qw, Q2w, nwPrinciple, n2wPrinciple,
    elabwNum, elab2wNum, a, PointGroup, DebugFlag,
    \[CurlyEpsilon]\[Omega]Cry, \[CurlyEpsilon]2\[Omega]Cry
  ];
  vars = case["mathematica_variables"];
  expected = case["expected_python_lab_inputs"];
  a = toComplex[vars["a"]];
  \[CurlyEpsilon]\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]\\[Omega]Cry"]];
  \[CurlyEpsilon]2\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]2\\[Omega]Cry"]];
  epsWLabExpected = toComplex[expected["epsilon_omega_lab"]];
  eps2WLabExpected = toComplex[expected["epsilon_2omega_lab"]];
  PointGroup = {"1"};
  DebugFlag = False;
  Internal`InheritedBlock[{Message},
    Unprotect[Message];
    Message[args___] := (
      AppendTo[messages, ToString[HoldForm[Message[args]], InputForm]];
      Null
    );
    Protect[Message];
    held = Quiet[ToExpression[principalSnippet, InputForm, HoldComplete]];
  ];
  If[held =!= $Failed, ReleaseHold[held]];
  <|
    "id" -> case["id"],
    "point_group_policy" -> "PointGroup set to {\"1\"} for general anisotropic principal-axis path",
    "parse_status" -> If[held === $Failed, "failed", "parsed"],
    "message_count" -> Length[messages],
    "messages_preview" -> Take[messages, UpTo[10]],
    "outputs" -> <|
      "a" -> jsonValue[a],
      "aCryToPrinw" -> jsonValue[aCryToPrinw],
      "aCryToPrin2w" -> jsonValue[aCryToPrin2w],
      "Qw" -> jsonValue[Qw],
      "Q2w" -> jsonValue[Q2w],
      "nwPrinciple" -> jsonValue[nwPrinciple],
      "n2wPrinciple" -> jsonValue[n2wPrinciple],
      "elabwNum" -> jsonValue[elabwNum],
      "elab2wNum" -> jsonValue[elab2wNum]
    |>,
    "checks" -> <|
      "max_abs_elabwNum_minus_expected_lab" -> N[maxAbs[elabwNum - epsWLabExpected], 17],
      "max_abs_elab2wNum_minus_expected_lab" -> N[maxAbs[elab2wNum - eps2WLabExpected], 17]
    |>
  |>
];

records = caseRecord /@ mapping["cases"];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.si principal-axis/index snippet stage reference",
    "status" -> "mathematica_stage_reference_exported",
    "validation_claim" -> "principal_axis_stage_only_not_full_shaarp_agreement",
    "wolfram_version" -> $Version,
    "mapping_path" -> mappingPath,
    "snippet_path" -> snippetPath,
    "region_id" -> "principal_axis_and_index_setup",
    "case_count" -> Length[records],
    "max_abs_elabwNum_minus_expected_lab" -> Max[records[[All, "checks", "max_abs_elabwNum_minus_expected_lab"]]],
    "max_abs_elab2wNum_minus_expected_lab" -> Max[records[[All, "checks", "max_abs_elab2wNum_minus_expected_lab"]]],
    "cases" -> records
  |>,
  "JSON"
];

Quit[]
