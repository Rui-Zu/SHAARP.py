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

(* Extract SHAARP.si's partial-analytical nonlinear-source polarizations
   (P2o, Peoo) LIVE from the SHAARP_V1.03.nb GUI cell via box-tree extraction,
   then evaluate them with a SYMBOLIC lab-frame d-tensor (dnewExp -> d11..d36)
   and SYMBOLIC field vectors, so we can check SHAARP.si's hand-written
   `e0*Simplify[{...}]` assembly against Python's compute_pnl_voigt_symbolic.

   P2o uses field ETwo (-> eo1,eo2,eo3); Peoo uses field ETweo (-> eeo1,eeo2,eeo3).
   Both are same-mode Voigt computePNL with an e0 (eps0) prefactor. *)

nbPath = SHAARPPaths`SI["SHAARP_V1.03/SHAARP_V1.03.nb"];
outputPath = SHAARPPaths`Ref["wolfram_live_symbolic_partial_pnl_reference_v1.json"];

nb = Import[nbPath, "NB"];
cells = Cases[nb, Cell[box_, "Input", ___] :> box, Infinity];
boxes = cells[[4]];

(* Symbolic substitutions used when releasing the held assignments. *)
dnewExp = Array[Symbol["d" <> ToString[#1] <> ToString[#2]] &, {3, 6}];
ETwo = {eo1, eo2, eo3};
ETweo = {eeo1, eeo2, eeo3};
e0 = 1;

(* Pull the minimal assignment sub-box for a given P symbol and evaluate it. *)
getAssign[name_String] := Module[{sym, asgnBoxes, held},
  sym = Symbol[name];
  asgnBoxes = Cases[
     boxes,
     b : RowBox[{name, "=", ___}] /; (StringLength[ToString[b, InputForm]] < 20000) :> b,
     Infinity
  ];
  If[Length[asgnBoxes] == 0, Return[$Failed]];
  held = ToExpression[First[asgnBoxes], StandardForm, HoldComplete];
  If[held === $Failed, Return[$Failed]];
  ReleaseHold[held];   (* assigns the symbol using current dnewExp/ETwo/ETweo/e0 *)
  sym
];

jsonExpr[expr_] := ToString[InputForm[expr]];

dRoles = Flatten[Table[("d" <> ToString[i] <> ToString[j]) -> ("Subscript[d," <> ToString[i] <> ToString[j] <> "]"),
   {i, 1, 3}, {j, 1, 6}]];
dSub1 = Flatten[Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> 0.07 i - 0.03 j, "imag" -> 0.011 i j|>, {i, 1, 3}, {j, 1, 6}]];
dSub2 = Flatten[Table[("d" <> ToString[i] <> ToString[j]) -> <|"real" -> -0.02 i + 0.05 j, "imag" -> -0.013 (i + j)|>, {i, 1, 3}, {j, 1, 6}]];

fieldSub1[name_, idx_] := name -> <|"real" -> 0.31 - 0.07 idx, "imag" -> -0.12 + 0.05 idx|>;
fieldSub2[name_, idx_] := name -> <|"real" -> -0.41 + 0.09 idx, "imag" -> 0.05 + 0.03 idx|>;

makeRecord[name_String, fieldNames_List] := Module[{val, flat, roles, s1, s2},
  val = getAssign[name];
  If[val === $Failed, Return[Null]];
  flat = Flatten[val];
  If[Length[flat] =!= 3, Return[Null]];
  roles = Association @@ Join[dRoles,
     (# -> "E_field_component" & ) /@ fieldNames];
  s1 = Association @@ Join[dSub1, MapIndexed[fieldSub1[#1, First[#2]] &, fieldNames]];
  s2 = Association @@ Join[dSub2, MapIndexed[fieldSub2[#1, First[#2]] &, fieldNames]];
  <|
    "expression_id" -> ("partial_pnl_" <> name),
    "shaarp_symbol" -> name,
    "source_stage" -> "SHAARP_si_partial_analytical_nonlinear_source",
    "output_component" -> "nonlinear_polarization_3vector",
    "expression_format" -> "mathematica_input_form",
    "expression" -> (jsonExpr /@ flat),
    "symbol_roles" -> roles,
    "numeric_substitutions" -> {s1, s2}
  |>
];

records = DeleteCases[{
    makeRecord["P2o", {"eo1", "eo2", "eo3"}],
    makeRecord["P2eo", {"eeo1", "eeo2", "eeo3"}],
    makeRecord["Peoo", {"eeo1", "eeo2", "eeo3", "eo1", "eo2", "eo3"}]
  }, Null];

payload = <|
  "source" -> "Wolfram 14.3 SHAARP.si SHAARP_V1.03.nb live-extracted partial-analytical nonlinear-source polarizations",
  "status" -> "mathematica_symbolic_reference_exported",
  "wolfram_version" -> $Version,
  "notebook_path" -> nbPath,
  "record_count" -> Length[records],
  "expressions" -> records
|>;

exportResult = Export[outputPath, payload, "JSON"];
Print["records = ", Length[records], " export = ", exportResult];
Quit[]
