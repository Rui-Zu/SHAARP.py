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

mappingPath = SHAARPPaths`Ref["shaarp_si_case_input_mapping_v1.json"];
snippetPath = SHAARPPaths`Ref["shaarp_si_refined_region_texts_v1.json"];
outputPath = SHAARPPaths`Ref["shaarp_si_pnl_stage_reference_v1.json"];

mapping = Import[mappingPath, "RawJSON"];
snippets = Import[snippetPath, "RawJSON"];
snippetById = Association[(#["region_id"] -> #["text"]) & /@ snippets["records"]];

toComplex[x_Association] /; KeyExistsQ[x, "real"] := N[x["real"] + I*x["imag"], 17];
toComplex[x_List] := toComplex /@ x;
toComplex[x_] := x;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
jsonValue[x_] := <|"input_form" -> ToString[x, InputForm]|>;

ruleRecord[rule_Rule] := <|"symbol" -> ToString[rule[[1]], InputForm], "value" -> jsonValue[rule[[2]]]|>;
ruleRecord[x_] := <|"input_form" -> ToString[x, InputForm]|>;

maxAbsOrMissing[x_] := Module[{n = Quiet[Check[N[Flatten[x]], $Failed]]},
  If[n === $Failed || !VectorQ[n, NumericQ], Missing["NonNumeric"], Max[Abs[n]]]
];

evalSnippet[id_] := Module[{held},
  held = Quiet[Check[ToExpression[snippetById[id], InputForm, HoldComplete], $Failed]];
  If[held =!= $Failed, ReleaseHold[held]];
  held =!= $Failed
];

applyWaveConstantBootstrap[] := (
  \[CurlyEpsilon]weoEff = nwNumExt^2;
  \[CurlyEpsilon]woEff = nwNumOrd^2;
  \[CurlyEpsilon]2weoEff = n2wNumExt^2;
  \[CurlyEpsilon]2woEff = n2wNumOrd^2;
  ktwo = w*Sqrt[\[CurlyEpsilon]woEff*e0];
  ktweo = w*Sqrt[\[CurlyEpsilon]weoEff*e0];
  kt2weo = 2*w*Sqrt[\[CurlyEpsilon]2weoEff*e0];
  kt2wo = 2*w*Sqrt[\[CurlyEpsilon]2woEff*e0];
  kr2w = 2*w*Sqrt[e0];
  kiw = w*Sqrt[e0];
  krw = w*Sqrt[e0];
);

assignDVariables[d_] := (
  {d11, d12, d13, d14, d15, d16} = d[[1]];
  {d21, d22, d23, d24, d25, d26} = d[[2]];
  {d31, d32, d33, d34, d35, d36} = d[[3]];
);

caseRecord[case_] := Module[
  {vars, dInput, ok, ewrootsRuleRecords, boundaryResiduals, maxBoundaryResidual, maxPNL},
  Clear[
    Inv, aCryToPrinw, aCryToPrin2w, Qw, Q2w, nwPrinciple, n2wPrinciple,
    elabwNum, elab2wNum, a, PointGroup, DebugFlag, ThetaIn, ThetaTout,
    \[CurlyEpsilon]\[Omega]Cry, \[CurlyEpsilon]2\[Omega]Cry,
    uy, ux, uz, L, RstandardwNum, Rstandard2wNum, CstandardwNum, Cstandard2wNum,
    AwNum, A2wNum, uMBy, uMBx, uMBz, LMB, RstandardMBwNum, RstandardMB2wNum,
    CstandardMBwNum, CstandardMB2wNum, AMBwNum, AMB2wNum,
    Vw, Ww, fo, fe, RootOrd, RootExt, ThetaOrd, ThetaExt, ExtwFactor,
    V2w, W2w, fo2w, fe2w, RootOrd2w, RootExt2w, ThetaOrd2w, ThetaExt2w, Ext2wFactor,
    nwNumExt, nwNumOrd, n2wNumExt, n2wNumOrd,
    EigenwNumExt, EigenwNumOrd, Eigen2wNumExt, Eigen2wNumOrd, k1Num, k2Num,
    ETwNumExt, ETwNumOrd, ET2wNumExt, ET2wNumOrd, DTwNumExt, DTwNumOrd,
    DT2wNumExt, DT2wNumOrd, VktwNumExt, VktwNumOrd, Vkt2wNumExt, Vkt2wNumOrd,
    HTwNumExt, HTwNumOrd, HT2wNumExt, HT2wNumOrd,
    \[CurlyEpsilon]weoEff, \[CurlyEpsilon]woEff, \[CurlyEpsilon]2weoEff, \[CurlyEpsilon]2woEff,
    ktwo, ktweo, kt2weo, kt2wo, kr2w, kiw, krw,
    RotatePolarizer, PolarizerAngle, Ellipticity, AnalyzerAngle, Eiw, EIw,
    \[CurlyPhi], \[Alpha], \[Beta], \[Gamma], RHTilt, RNum, \[Mu], \[Mu]0, e0, w,
    EIwNum, DIwNum, VkiwNum, ERpNum, ERsNum, ERwNum, DRwNum, VkrwNum, HIwNum, HRwNum,
    EwrootsNum, twe, two, ETweo, ETwo, Vktweo, Vktwo, elab2w,
    Isotropic, Uniaxial, Biaxial,
    d11, d12, d13, d14, d15, d16, d21, d22, d23, d24, d25, d26,
    d31, d32, d33, d34, d35, d36, dold, T, dSHG, i, j, k, l, m, n,
    P2eo, P2o, Peoo
  ];
  vars = case["mathematica_variables"];
  a = toComplex[vars["a"]];
  \[CurlyEpsilon]\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]\\[Omega]Cry"]];
  \[CurlyEpsilon]2\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]2\\[Omega]Cry"]];
  dInput = toComplex[vars["dold"]];
  ThetaIn = vars["ThetaIn"];
  PolarizerAngle = vars["PolarizerAngle"];
  Ellipticity = vars["Ellipticity"];
  AnalyzerAngle = vars["AnalyzerAngle"];
  RotatePolarizer = False;
  PointGroup = {"1"};
  DebugFlag = False;
  Isotropic = False;
  Uniaxial = False;
  Biaxial = True;
  \[Mu] = IdentityMatrix[3];
  \[Mu]0 = 1;
  e0 = 1;
  w = 1;
  assignDVariables[dInput];
  ok = And[
    evalSnippet["principal_axis_and_index_setup"],
    evalSnippet["active_wave_equation_setup"],
    evalSnippet["active_backward_wave_equation_setup"],
    evalSnippet["omega_angle_root_assignments_active_path"],
    evalSnippet["two_omega_angle_root_assignments_active_path"],
    evalSnippet["omega_angle_roots_active_numerical_path"]
  ];
  applyWaveConstantBootstrap[];
  ok = And[
    ok,
    evalSnippet["omega_transmitted_field_directions_active_path"],
    evalSnippet["omega_incident_controls_active_path"],
    evalSnippet["omega_fresnel_field_setup_active_path"],
    evalSnippet["omega_fresnel_equations_active_path"]
  ];
  ewrootsRuleRecords = If[ok && ListQ[EwrootsNum] && Length[EwrootsNum] > 0, ruleRecord /@ EwrootsNum[[1]], {}];
  ok = And[
    ok,
    evalSnippet["omega_fresnel_substitution_active_path"],
    evalSnippet["shg_tensor_input_matrix"],
    evalSnippet["shg_tensor_transformation_active_path"],
    evalSnippet["active_complex_field_pnl_block"]
  ];
  boundaryResiduals = Quiet[N[{
    EIwNum[[1]] + ERwNum[[1]] - (ETwNumOrd[[1]] + ETwNumExt[[1]]),
    EIwNum[[2]] + ERwNum[[2]] - (ETwNumOrd[[2]] + ETwNumExt[[2]]),
    HIwNum[[1]] + HRwNum[[1]] - (HTwNumOrd[[1]] + HTwNumExt[[1]]),
    HIwNum[[2]] + HRwNum[[2]] - (HTwNumOrd[[2]] + HTwNumExt[[2]])
  }]];
  maxBoundaryResidual = maxAbsOrMissing[boundaryResiduals];
  maxPNL = maxAbsOrMissing[{P2eo, P2o, Peoo}];
  <|
    "id" -> case["id"],
    "parse_status" -> If[ok, "parsed", "failed"],
    "policy" -> <|
      "anisotropy_branch_policy" -> "Biaxial=True, Isotropic=False, Uniaxial=False for the complex anisotropic solver-stage benchmark family",
      "wave_constant_policy" -> "ktwo/ktweo/kt2wo/kt2weo/kiw/krw are bootstrapped before transmitted-field snippets using SHAARP.si later numerical formulas from lines 850-856",
      "dold_policy" -> "d11...d36 assigned from case mapping, then SHAARP.si shg_tensor_input_matrix snippet reconstructs dold"
    |>,
    "outputs" -> <|
      "EwrootsNum" -> ewrootsRuleRecords,
      "dold" -> jsonValue[dold],
      "dSHG" -> jsonValue[dSHG],
      "ETweo" -> jsonValue[ETweo],
      "ETwo" -> jsonValue[ETwo],
      "P2eo" -> jsonValue[P2eo],
      "P2o" -> jsonValue[P2o],
      "Peoo" -> jsonValue[Peoo]
    |>,
    "checks" -> <|
      "max_linear_boundary_abs_residual" -> If[Head[maxBoundaryResidual] === Missing, ToString[maxBoundaryResidual, InputForm], N[maxBoundaryResidual, 17]],
      "max_abs_pnl_component" -> If[Head[maxPNL] === Missing, ToString[maxPNL, InputForm], N[maxPNL, 17]]
    |>
  |>
];

records = caseRecord /@ mapping["cases"];
numericBoundaryResiduals = Select[records[[All, "checks", "max_linear_boundary_abs_residual"]], NumericQ];
numericPNLMaxima = Select[records[[All, "checks", "max_abs_pnl_component"]], NumericQ];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.si active complex PNL snippet stage reference",
    "status" -> "mathematica_stage_reference_exported",
    "validation_claim" -> "pnl_stage_only_not_full_shaarp_agreement",
    "wolfram_version" -> $Version,
    "mapping_path" -> mappingPath,
    "snippet_path" -> snippetPath,
    "region_ids" -> {
      "principal_axis_and_index_setup",
      "active_wave_equation_setup",
      "active_backward_wave_equation_setup",
      "omega_angle_root_assignments_active_path",
      "two_omega_angle_root_assignments_active_path",
      "omega_angle_roots_active_numerical_path",
      "omega_transmitted_field_directions_active_path",
      "omega_incident_controls_active_path",
      "omega_fresnel_field_setup_active_path",
      "omega_fresnel_equations_active_path",
      "omega_fresnel_substitution_active_path",
      "shg_tensor_input_matrix",
      "shg_tensor_transformation_active_path",
      "active_complex_field_pnl_block"
    },
    "policy" -> <|
      "anisotropy_branch_policy" -> "Biaxial=True, Isotropic=False, Uniaxial=False for all exported cases",
      "wave_constant_policy" -> "Explicit pre-Fresnel bootstrap from SHAARP.si line-850-to-856 formulas because the audited PlainText order uses ktwo/ktweo before those later assignments",
      "dold_policy" -> "d11...d36 assigned from shaarp_si_case_input_mapping_v1.json, then SHAARP.si input-matrix snippet rebuilds dold before dSHG rotation"
    |>,
    "case_count" -> Length[records],
    "numeric_boundary_residual_case_count" -> Length[numericBoundaryResiduals],
    "numeric_pnl_case_count" -> Length[numericPNLMaxima],
    "max_linear_boundary_abs_residual" -> If[Length[numericBoundaryResiduals] == 0, "non_numeric", Max[numericBoundaryResiduals]],
    "max_abs_pnl_component" -> If[Length[numericPNLMaxima] == 0, "non_numeric", Max[numericPNLMaxima]],
    "cases" -> records
  |>,
  "JSON"
];

Quit[]
