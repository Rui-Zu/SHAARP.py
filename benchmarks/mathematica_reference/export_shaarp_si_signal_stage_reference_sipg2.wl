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

mappingPath = SHAARPPaths`Ref["shaarp_si_case_input_mapping_sipg2.json"];
snippetPath = SHAARPPaths`Ref["shaarp_si_refined_region_texts_v1.json"];
outputPath = SHAARPPaths`Ref["shaarp_si_signal_stage_reference_sipg2.json"];

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
namedValueRecord[name_, value_] := <|"symbol" -> name, "value" -> jsonValue[value]|>;

scalarOne[x_List] /; Length[x] == 1 := First[x];
scalarOne[x_] := x;

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
  {
    vars, dInput, ok, ewrootsRuleRecords, crootsRuleRecords,
    boundaryResiduals, maxBoundaryResidual, maxPNL, cResiduals, maxCResidual,
    sourceMaxima, maxSourceField, e2wrootsRuleRecords, shBoundaryResiduals,
    maxSHBoundaryResidual, maxSHSignal, maxIntensityValue
  },
  Clear[
    Inv, aCryToPrinw, aCryToPrin2w, Qw, Q2w, nwPrinciple, n2wPrinciple,
    elabwNum, elab2wNum, a, PointGroup, DebugFlag, ThetaIn, ThetaTout,
    \[CurlyEpsilon]\[Omega]Cry, \[CurlyEpsilon]2\[Omega]Cry,
    uy, ux, uz, L, RstandardwNum, Rstandard2wNum, CstandardwNum, Cstandard2wNum,
    AwNum, A2wNum, uMBy, uMBx, uMBz, LMB, RstandardMBwNum, RstandardMB2wNum,
    CstandardMBwNum, CstandardMB2wNum, AMBwNum, AMB2wNum,
    Vw, Ww, fo, fe, RootOrd, RootExt, ThetaOrd, ThetaExt, ExtwFactor,
    V2w, W2w, fo2w, fe2w, RootOrd2w, RootExt2w, ThetaOrd2w, ThetaExt2w, Ext2wFactor,
    nwNumExt, nwNumOrd, n2wNumExt, n2wNumOrd, nMB2wNumExt, nMB2wNumOrd,
    EigenwNumExt, EigenwNumOrd, Eigen2wNumExt, Eigen2wNumOrd, k1Num, k2Num,
    ETwNumExt, ETwNumOrd, ET2wNumExt, ET2wNumOrd, DTwNumExt, DTwNumOrd,
    DT2wNumExt, DT2wNumOrd, VktwNumExt, VktwNumOrd, Vkt2wNumExt, Vkt2wNumOrd,
    HTwNumExt, HTwNumOrd, HT2wNumExt, HT2wNumOrd,
    \[CurlyEpsilon]weoEff, \[CurlyEpsilon]woEff, \[CurlyEpsilon]2weoEff, \[CurlyEpsilon]2woEff,
    ktwo, ktweo, kt2weo, kt2wo, kr2w, kiw, krw, ktMB2wo, ktMB2weo,
    neMB2w, noMB2w,
    RotatePolarizer, RotateAnalyzer, PolarizerAngle, Ellipticity, AnalyzerAngle, Eiw, EIw,
    \[CurlyPhi], \[Alpha], \[Beta], \[Gamma], RHTilt, RNum, \[Mu], \[Mu]0, e0, w,
    EIwNum, DIwNum, VkiwNum, ERpNum, ERsNum, ERwNum, DRwNum, VkrwNum, HIwNum, HRwNum,
    EwrootsNum, twe, two, ETweo, ETwo, Vktweo, Vktwo, elab2w,
    Isotropic, Uniaxial, Biaxial,
    d11, d12, d13, d14, d15, d16, d21, d22, d23, d24, d25, d26,
    d31, d32, d33, d34, d35, d36, dold, T, dSHG, i, j, k, l, m, n,
    P2eo, P2o, Peoo,
    r, t, x, y, z, g2o, g2eo, geoo, Lg2o, Lg2eo, Lgeoo, R2o, R2eo, Reoo,
    C11, C12, C13, C21, C22, C23, C31, C32, C33, Croots,
    ER2wp, ER2ws, ER2w, DR2w, Vkr2w, HR2w,
    RC11, RC12, RC13, RC21, RC22, RC23, RC31, RC32, RC33,
    DT2w2eo, Vkt2w2eo, DT2w2o, Vkt2w2o, DT2weoo, Vkt2weoo,
    HT2w2o, HT2w2eo, HT2weoo,
    E2wroots, ET2weoA, ET2woA, RSignal,
    RSignalCrystal, I2wx, I2wy, I2wParallel, I2wPerpdicular,
    MaxI2wx, MaxI2wy, MaxI2w
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
  RotateAnalyzer = False;
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
    evalSnippet["active_complex_field_pnl_block"],
    evalSnippet["inhomogeneous_particular_solution_coefficients"]
  ];
  crootsRuleRecords = If[ok && ListQ[Croots] && Length[Croots] > 0, ruleRecord /@ Croots[[1]], {}];
  cResiduals = Quiet[Check[N[Flatten[{Lg2o - R2o, Lg2eo - R2eo, Lgeoo - Reoo} /. Croots[[1]]]], $Failed]];
  ok = And[ok, evalSnippet["numerical_coefficients_and_two_omega_field_setup"]];
  boundaryResiduals = Quiet[N[{
    EIwNum[[1]] + ERwNum[[1]] - (ETwNumOrd[[1]] + ETwNumExt[[1]]),
    EIwNum[[2]] + ERwNum[[2]] - (ETwNumOrd[[2]] + ETwNumExt[[2]]),
    HIwNum[[1]] + HRwNum[[1]] - (HTwNumOrd[[1]] + HTwNumExt[[1]]),
    HIwNum[[2]] + HRwNum[[2]] - (HTwNumOrd[[2]] + HTwNumExt[[2]])
  }]];
  maxBoundaryResidual = maxAbsOrMissing[boundaryResiduals];
  maxPNL = maxAbsOrMissing[{P2eo, P2o, Peoo}];
  maxCResidual = maxAbsOrMissing[cResiduals];
  sourceMaxima = {
    RC11, RC12, RC13, RC21, RC22, RC23, RC31, RC32, RC33,
    DT2w2eo, DT2w2o, DT2weoo,
    Vkt2w2eo, Vkt2w2o, Vkt2weoo,
    HT2w2eo, HT2w2o, HT2weoo
  };
  maxSourceField = maxAbsOrMissing[sourceMaxima];
  ok = And[ok, evalSnippet["two_omega_boundary_solution"]];
  e2wrootsRuleRecords = If[
    ok,
    {
      namedValueRecord["ER2wp", ER2wp],
      namedValueRecord["ER2ws", ER2ws],
      namedValueRecord["ET2weoA", ET2weoA],
      namedValueRecord["ET2woA", ET2woA]
    },
    {}
  ];
  shBoundaryResiduals = Quiet[N[{
    ER2w[[1]] - (ET2weoA*ET2wNumExt[[1]] + ET2woA*ET2wNumOrd[[1]] + RC11 + RC12 + RC13),
    ER2w[[2]] - (ET2weoA*ET2wNumExt[[2]] + ET2woA*ET2wNumOrd[[2]] + RC21 + RC22 + RC23),
    HR2w[[1]] - (ET2weoA*HT2wNumExt[[1]] + ET2woA*HT2wNumOrd[[1]] + HT2w2eo[[1]] + HT2w2o[[1]] + HT2weoo[[1]]),
    HR2w[[2]] - (ET2weoA*HT2wNumExt[[2]] + ET2woA*HT2wNumOrd[[2]] + HT2w2eo[[2]] + HT2w2o[[2]] + HT2weoo[[2]])
  }]];
  maxSHBoundaryResidual = maxAbsOrMissing[shBoundaryResiduals];
  maxSHSignal = maxAbsOrMissing[{ER2wp, ER2ws, ET2weoA, ET2woA, RSignal, ER2w, HR2w}];
  ok = And[ok, evalSnippet["reflected_signal_and_basic_intensity"]];
  maxIntensityValue = maxAbsOrMissing[
    {I2wx[0], I2wy[0], I2wParallel[0], I2wPerpdicular[0], MaxI2wx, MaxI2wy, MaxI2w}
  ];
  <|
    "id" -> case["id"],
    "parse_status" -> If[ok, "parsed", "failed"],
    "policy" -> <|
      "anisotropy_branch_policy" -> "Biaxial=True, Isotropic=False, Uniaxial=False for the complex anisotropic solver-stage benchmark family",
      "wave_constant_policy" -> "numerical_coefficients_and_two_omega_field_setup re-executes SHAARP.si line-850-to-856 wave-constant assignments after Croots",
      "boundary_unknown_policy" -> "ER2wp and ER2ws are solved by SHAARP.si two_omega_boundary_solution before this signal stage",
      "rotate_analyzer_policy" -> "RotateAnalyzer=False for these fixed-analyzer benchmark cases"
    |>,
    "outputs" -> <|
      "EwrootsNum" -> ewrootsRuleRecords,
      "Croots" -> crootsRuleRecords,
      "E2wroots" -> e2wrootsRuleRecords,
      "RC" -> jsonValue[{
        {scalarOne[RC11], scalarOne[RC12], scalarOne[RC13]},
        {scalarOne[RC21], scalarOne[RC22], scalarOne[RC23]},
        {scalarOne[RC31], scalarOne[RC32], scalarOne[RC33]}
      }],
      "DT2w2eo" -> jsonValue[DT2w2eo],
      "DT2w2o" -> jsonValue[DT2w2o],
      "DT2weoo" -> jsonValue[DT2weoo],
      "Vkt2w2eo" -> jsonValue[Vkt2w2eo],
      "Vkt2w2o" -> jsonValue[Vkt2w2o],
      "Vkt2weoo" -> jsonValue[Vkt2weoo],
      "HT2w2eo" -> jsonValue[HT2w2eo],
      "HT2w2o" -> jsonValue[HT2w2o],
      "HT2weoo" -> jsonValue[HT2weoo],
      "ET2wNumExt" -> jsonValue[ET2wNumExt],
      "ET2wNumOrd" -> jsonValue[ET2wNumOrd],
      "HT2wNumExt" -> jsonValue[HT2wNumExt],
      "HT2wNumOrd" -> jsonValue[HT2wNumOrd],
      "ER2wp" -> jsonValue[ER2wp],
      "ER2ws" -> jsonValue[ER2ws],
      "ET2weoA" -> jsonValue[ET2weoA],
      "ET2woA" -> jsonValue[ET2woA],
      "RSignal" -> jsonValue[RSignal],
      "ER2w" -> jsonValue[ER2w],
      "HR2w" -> jsonValue[HR2w],
      "RSignalCrystal" -> jsonValue[RSignalCrystal],
      "I2wx_at_phi_0" -> jsonValue[I2wx[0]],
      "I2wy_at_phi_0" -> jsonValue[I2wy[0]],
      "I2wParallel_at_phi_0" -> jsonValue[I2wParallel[0]],
      "I2wPerpdicular_at_phi_0" -> jsonValue[I2wPerpdicular[0]],
      "MaxI2wx" -> jsonValue[MaxI2wx],
      "MaxI2wy" -> jsonValue[MaxI2wy],
      "MaxI2w" -> jsonValue[MaxI2w]
    |>,
    "checks" -> <|
      "max_linear_boundary_abs_residual" -> If[Head[maxBoundaryResidual] === Missing, ToString[maxBoundaryResidual, InputForm], N[maxBoundaryResidual, 17]],
      "max_abs_pnl_component" -> If[Head[maxPNL] === Missing, ToString[maxPNL, InputForm], N[maxPNL, 17]],
      "max_croots_abs_residual" -> If[Head[maxCResidual] === Missing, ToString[maxCResidual, InputForm], N[maxCResidual, 17]],
      "max_abs_two_omega_setup_component" -> If[Head[maxSourceField] === Missing, ToString[maxSourceField, InputForm], N[maxSourceField, 17]],
      "max_two_omega_boundary_abs_residual" -> If[Head[maxSHBoundaryResidual] === Missing, ToString[maxSHBoundaryResidual, InputForm], N[maxSHBoundaryResidual, 17]],
      "max_abs_two_omega_signal_component" -> If[Head[maxSHSignal] === Missing, ToString[maxSHSignal, InputForm], N[maxSHSignal, 17]],
      "max_abs_intensity_component" -> If[Head[maxIntensityValue] === Missing, ToString[maxIntensityValue, InputForm], N[maxIntensityValue, 17]]
    |>
  |>
];

records = caseRecord /@ mapping["cases"];
numericBoundaryResiduals = Select[records[[All, "checks", "max_linear_boundary_abs_residual"]], NumericQ];
numericPNLMaxima = Select[records[[All, "checks", "max_abs_pnl_component"]], NumericQ];
numericCrootsResiduals = Select[records[[All, "checks", "max_croots_abs_residual"]], NumericQ];
numericSetupMaxima = Select[records[[All, "checks", "max_abs_two_omega_setup_component"]], NumericQ];
numericSHBoundaryResiduals = Select[records[[All, "checks", "max_two_omega_boundary_abs_residual"]], NumericQ];
numericSHSignalMaxima = Select[records[[All, "checks", "max_abs_two_omega_signal_component"]], NumericQ];
numericIntensityMaxima = Select[records[[All, "checks", "max_abs_intensity_component"]], NumericQ];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.si reflected signal and basic intensity snippet stage reference",
    "status" -> "mathematica_stage_reference_exported",
    "validation_claim" -> "signal_stage_only_not_full_shaarp_agreement",
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
      "active_complex_field_pnl_block",
      "inhomogeneous_particular_solution_coefficients",
      "numerical_coefficients_and_two_omega_field_setup",
      "two_omega_boundary_solution",
      "reflected_signal_and_basic_intensity"
    },
    "policy" -> <|
      "anisotropy_branch_policy" -> "Biaxial=True, Isotropic=False, Uniaxial=False for all exported cases",
      "boundary_unknown_policy" -> "ER2wp, ER2ws, ET2weoA, and ET2woA are exported with explicit names after SHAARP.si overwrites those symbols with solved scalar values",
      "rotate_analyzer_policy" -> "RotateAnalyzer=False for fixed analyzer-angle benchmark cases"
    |>,
    "case_count" -> Length[records],
    "numeric_boundary_residual_case_count" -> Length[numericBoundaryResiduals],
    "numeric_pnl_case_count" -> Length[numericPNLMaxima],
    "numeric_croots_residual_case_count" -> Length[numericCrootsResiduals],
    "numeric_two_omega_setup_case_count" -> Length[numericSetupMaxima],
    "numeric_two_omega_boundary_residual_case_count" -> Length[numericSHBoundaryResiduals],
    "numeric_two_omega_signal_case_count" -> Length[numericSHSignalMaxima],
    "numeric_intensity_case_count" -> Length[numericIntensityMaxima],
    "max_linear_boundary_abs_residual" -> If[Length[numericBoundaryResiduals] == 0, "non_numeric", Max[numericBoundaryResiduals]],
    "max_abs_pnl_component" -> If[Length[numericPNLMaxima] == 0, "non_numeric", Max[numericPNLMaxima]],
    "max_croots_abs_residual" -> If[Length[numericCrootsResiduals] == 0, "non_numeric", Max[numericCrootsResiduals]],
    "max_abs_two_omega_setup_component" -> If[Length[numericSetupMaxima] == 0, "non_numeric", Max[numericSetupMaxima]],
    "max_two_omega_boundary_abs_residual" -> If[Length[numericSHBoundaryResiduals] == 0, "non_numeric", Max[numericSHBoundaryResiduals]],
    "max_abs_two_omega_signal_component" -> If[Length[numericSHSignalMaxima] == 0, "non_numeric", Max[numericSHSignalMaxima]],
    "max_abs_intensity_component" -> If[Length[numericIntensityMaxima] == 0, "non_numeric", Max[numericIntensityMaxima]],
    "cases" -> records
  |>,
  "JSON"
];

Quit[]
