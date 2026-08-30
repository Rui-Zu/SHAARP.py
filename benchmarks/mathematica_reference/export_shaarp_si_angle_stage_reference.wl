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
outputPath = SHAARPPaths`Ref["shaarp_si_angle_stage_reference_v1.json"];

mapping = Import[mappingPath, "RawJSON"];
snippets = Import[snippetPath, "RawJSON"];
snippetById = Association[(#["region_id"] -> #["text"]) & /@ snippets["records"]];

toComplex[x_Association] /; KeyExistsQ[x, "real"] := N[x["real"] + I*x["imag"], 17];
toComplex[x_List] := toComplex /@ x;
toComplex[x_] := x;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;

maxAbs[x_] := Max[Abs[Flatten[N[x]]]];

evalSnippet[id_, messages_] := Module[{held},
  held = Quiet[Check[ToExpression[snippetById[id], InputForm, HoldComplete], $Failed]];
  If[held =!= $Failed, ReleaseHold[held]];
  held =!= $Failed
];

caseRecord[case_] := Module[
  {vars, messages = {}, ok, omegaOrdResidual, omegaExtResidual, twoOmegaOrdResidual, twoOmegaExtResidual},
  Clear[
    Inv, aCryToPrinw, aCryToPrin2w, Qw, Q2w, nwPrinciple, n2wPrinciple,
    elabwNum, elab2wNum, a, PointGroup, DebugFlag, ThetaIn, ThetaTout,
    \[CurlyEpsilon]\[Omega]Cry, \[CurlyEpsilon]2\[Omega]Cry,
    uy, ux, uz, L, RstandardwNum, Rstandard2wNum, CstandardwNum, Cstandard2wNum,
    AwNum, A2wNum, uMBy, uMBx, uMBz, LMB, RstandardMBwNum, RstandardMB2wNum,
    CstandardMBwNum, CstandardMB2wNum, AMBwNum, AMB2wNum,
    Vw, Ww, fo, fe, RootOrd, RootExt, ThetaOrd, ThetaExt, ExtwFactor,
    V2w, W2w, fo2w, fe2w, RootOrd2w, RootExt2w, ThetaOrd2w, ThetaExt2w, Ext2wFactor,
    nwNumExt, nwNumOrd, n2wNumExt, n2wNumOrd
  ];
  vars = case["mathematica_variables"];
  a = toComplex[vars["a"]];
  \[CurlyEpsilon]\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]\\[Omega]Cry"]];
  \[CurlyEpsilon]2\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]2\\[Omega]Cry"]];
  ThetaIn = vars["ThetaIn"];
  PointGroup = {"1"};
  DebugFlag = False;
  ok = And[
    evalSnippet["principal_axis_and_index_setup", messages],
    evalSnippet["active_wave_equation_setup", messages],
    evalSnippet["active_backward_wave_equation_setup", messages],
    evalSnippet["omega_angle_root_assignments_active_path", messages],
    evalSnippet["two_omega_angle_root_assignments_active_path", messages],
    evalSnippet["omega_angle_roots_active_numerical_path", messages]
  ];
  omegaOrdResidual = Quiet[Abs[N[fo /. ThetaTout -> ThetaOrd]]];
  omegaExtResidual = Quiet[Abs[N[fe /. ThetaTout -> ThetaExt]]];
  twoOmegaOrdResidual = Quiet[Abs[N[fo2w /. ThetaTout -> ThetaOrd2w]]];
  twoOmegaExtResidual = Quiet[Abs[N[fe2w /. ThetaTout -> ThetaExt2w]]];
  <|
    "id" -> case["id"],
    "parse_status" -> If[ok, "parsed", "failed"],
    "message_count" -> Length[messages],
    "messages_preview" -> Take[messages, UpTo[10]],
    "outputs" -> <|
      "ThetaOrd" -> jsonValue[ThetaOrd],
      "ThetaExt" -> jsonValue[ThetaExt],
      "ThetaOrd2w" -> jsonValue[ThetaOrd2w],
      "ThetaExt2w" -> jsonValue[ThetaExt2w],
      "nwNumOrd" -> jsonValue[nwNumOrd],
      "nwNumExt" -> jsonValue[nwNumExt],
      "n2wNumOrd" -> jsonValue[n2wNumOrd],
      "n2wNumExt" -> jsonValue[n2wNumExt],
      "ExtwFactor" -> jsonValue[ExtwFactor],
      "Ext2wFactor" -> jsonValue[Ext2wFactor]
    |>,
    "checks" -> <|
      "omega_ord_root_abs_residual" -> N[omegaOrdResidual, 17],
      "omega_ext_root_abs_residual" -> N[omegaExtResidual, 17],
      "two_omega_ord_root_abs_residual" -> N[twoOmegaOrdResidual, 17],
      "two_omega_ext_root_abs_residual" -> N[twoOmegaExtResidual, 17]
    |>
  |>
];

records = caseRecord /@ mapping["cases"];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.si active angle-root snippet stage reference",
    "status" -> "mathematica_stage_reference_exported",
    "validation_claim" -> "angle_stage_only_not_full_shaarp_agreement",
    "wolfram_version" -> $Version,
    "mapping_path" -> mappingPath,
    "snippet_path" -> snippetPath,
    "region_ids" -> {
      "principal_axis_and_index_setup",
      "active_wave_equation_setup",
      "active_backward_wave_equation_setup",
      "omega_angle_root_assignments_active_path",
      "two_omega_angle_root_assignments_active_path",
      "omega_angle_roots_active_numerical_path"
    },
    "case_count" -> Length[records],
    "max_omega_ord_root_abs_residual" -> Max[records[[All, "checks", "omega_ord_root_abs_residual"]]],
    "max_omega_ext_root_abs_residual" -> Max[records[[All, "checks", "omega_ext_root_abs_residual"]]],
    "max_two_omega_ord_root_abs_residual" -> Max[records[[All, "checks", "two_omega_ord_root_abs_residual"]]],
    "max_two_omega_ext_root_abs_residual" -> Max[records[[All, "checks", "two_omega_ext_root_abs_residual"]]],
    "cases" -> records
  |>,
  "JSON"
];

Quit[]
