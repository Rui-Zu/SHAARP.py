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
outputPath = SHAARPPaths`Ref["shaarp_si_branch_metadata_reference_v1.json"];

mapping = Import[mappingPath, "RawJSON"];
snippets = Import[snippetPath, "RawJSON"];
snippetById = Association[(#["region_id"] -> #["text"]) & /@ snippets["records"]];

toComplex[x_Association] /; KeyExistsQ[x, "real"] := N[x["real"] + I*x["imag"], 17];
toComplex[x_List] := toComplex /@ x;
toComplex[x_] := x;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
jsonValue[x_] := <|"input_form" -> ToString[x, InputForm]|>;

evalSnippet[id_] := Module[{held},
  held = Quiet[Check[ToExpression[snippetById[id], InputForm, HoldComplete], $Failed]];
  If[held =!= $Failed, ReleaseHold[held]];
  held =!= $Failed
];

indexFromEigenValues[eigenValues_, refractiveIndex_] := Module[
  {roundedEigenIndices},
  roundedEigenIndices = Flatten[Position[
    SetPrecision[Sqrt[1 / eigenValues], 6],
    SetPrecision[refractiveIndex, 6]
  ]];
  If[Length[roundedEigenIndices] > 0, First[roundedEigenIndices], Missing["NoRoundedIndexMatch"]]
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

caseRecord[case_] := Module[
  {vars, ok, omegaExtIndex, omegaOrdIndex, twoOmegaExtIndex, twoOmegaOrdIndex},
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
    RotatePolarizer, RotateAnalyzer, PolarizerAngle, Ellipticity, AnalyzerAngle, Eiw, EIw,
    \[CurlyPhi], \[Alpha], \[Beta], \[Gamma], RHTilt, RNum, \[Mu], \[Mu]0, e0, w,
    EIwNum, DIwNum, VkiwNum, ERpNum, ERsNum, ERwNum, DRwNum, VkrwNum, HIwNum, HRwNum,
    EwrootsNum, twe, two, ETweo, ETwo, Vktweo, Vktwo, elab2w,
    Isotropic, Uniaxial, Biaxial
  ];
  vars = case["mathematica_variables"];
  a = toComplex[vars["a"]];
  \[CurlyEpsilon]\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]\\[Omega]Cry"]];
  \[CurlyEpsilon]2\[Omega]Cry = toComplex[vars["\\[CurlyEpsilon]2\\[Omega]Cry"]];
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
    evalSnippet["omega_transmitted_field_directions_active_path"]
  ];
  omegaExtIndex = indexFromEigenValues[EigenwNumExt[[1]], nwNumExt];
  omegaOrdIndex = indexFromEigenValues[EigenwNumOrd[[1]], nwNumOrd];
  twoOmegaExtIndex = indexFromEigenValues[Eigen2wNumExt[[1]], n2wNumExt];
  twoOmegaOrdIndex = indexFromEigenValues[Eigen2wNumOrd[[1]], n2wNumOrd];
  <|
    "id" -> case["id"],
    "parse_status" -> If[ok, "parsed", "failed"],
    "outputs" -> <|
      "omega_ext_k_index" -> omegaExtIndex,
      "omega_ord_k_index" -> omegaOrdIndex,
      "two_omega_ext_k_index" -> twoOmegaExtIndex,
      "two_omega_ord_k_index" -> twoOmegaOrdIndex,
      "EigenwNumExt_values" -> jsonValue[EigenwNumExt[[1]]],
      "EigenwNumExt_vectors" -> jsonValue[EigenwNumExt[[2]]],
      "EigenwNumOrd_values" -> jsonValue[EigenwNumOrd[[1]]],
      "EigenwNumOrd_vectors" -> jsonValue[EigenwNumOrd[[2]]],
      "Eigen2wNumExt_values" -> jsonValue[Eigen2wNumExt[[1]]],
      "Eigen2wNumExt_vectors" -> jsonValue[Eigen2wNumExt[[2]]],
      "Eigen2wNumOrd_values" -> jsonValue[Eigen2wNumOrd[[1]]],
      "Eigen2wNumOrd_vectors" -> jsonValue[Eigen2wNumOrd[[2]]],
      "ETwNumExt" -> jsonValue[ETwNumExt],
      "ETwNumOrd" -> jsonValue[ETwNumOrd],
      "ET2wNumExt" -> jsonValue[ET2wNumExt],
      "ET2wNumOrd" -> jsonValue[ET2wNumOrd],
      "VktwNumExt" -> jsonValue[VktwNumExt],
      "VktwNumOrd" -> jsonValue[VktwNumOrd],
      "Vkt2wNumExt" -> jsonValue[Vkt2wNumExt],
      "Vkt2wNumOrd" -> jsonValue[Vkt2wNumOrd]
    |>
  |>
];

records = caseRecord /@ mapping["cases"];
numericRecords = Select[
  records,
  And[
    NumericQ[#["outputs", "omega_ext_k_index"]],
    NumericQ[#["outputs", "omega_ord_k_index"]],
    NumericQ[#["outputs", "two_omega_ext_k_index"]],
    NumericQ[#["outputs", "two_omega_ord_k_index"]]
  ] &
];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.si active eigensystem branch metadata reference",
    "status" -> "mathematica_stage_reference_exported",
    "validation_claim" -> "branch_metadata_stage_only_not_full_shaarp_agreement",
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
      "omega_transmitted_field_directions_active_path"
    },
    "case_count" -> Length[records],
    "numeric_branch_case_count" -> Length[numericRecords],
    "cases" -> records
  |>,
  "JSON"
];

Quit[]
