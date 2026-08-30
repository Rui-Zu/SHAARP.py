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

flagLinearSolve = False;
flagNSolve = True;
debug = False;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;

omegaKey = \[Omega];
eps2OmegaLKey = ToExpression["Global`" <> FromCharacterCode[{949, 50, 969, 76}, "Unicode"]];

makeD[seed_] := Table[
  N[(0.06*seed + 0.09*i - 0.04*j) + I*(0.035*seed - 0.025*i + 0.045*j)],
  {i, 1, 3},
  {j, 1, 6}
];

makeE[seed_, offset_] := Table[
  N[(0.17*seed - 0.11*i + 0.05*offset) + I*(0.09*seed + 0.04*i - 0.025*offset)],
  {i, 1, 3}
];

makeEpsilon[seed_] := Module[{base},
  base = {
    {2.2 + 0.07*seed + I*(0.04 + 0.006*seed), 0.035 + 0.004*seed - 0.018*I, -0.025 + 0.003*seed + 0.021*I},
    {0.028 - 0.002*seed + 0.026*I, 2.6 + 0.05*seed + I*(0.035 + 0.005*seed), 0.045 - 0.012*I},
    {-0.018 + 0.017*I, 0.033 + 0.002*seed + 0.019*I, 3.0 + 0.06*seed + I*(0.03 + 0.004*seed)}
  };
  N[base]
];

makeCase[idx_] := Module[
  {omega, d, eps2, k1, k2, e1, e2, matCase, w1Case, w2Case, w11Case, w22Case, wmixCase},
  Clear[matCase, w1Case, w2Case, w11Case, w22Case, wmixCase, C11, C21, C31, C12, C22, C32, C13, C23, C33];
  omega = N[1.0 + 0.035*idx];
  d = makeD[idx];
  eps2 = makeEpsilon[idx];
  k1 = N[{0.05 + 0.014*idx, 0., 0.8 + 0.033*idx + 0.01*I*Mod[idx, 3]}];
  k2 = N[{0.05 + 0.014*idx, 0., 1.0 + 0.027*idx + 0.008*I*Mod[idx + 1, 4]}];
  e1 = makeE[idx, 1];
  e2 = makeE[idx + 2, 2];
  With[{ok = omegaKey, ek = eps2OmegaLKey},
    matCase[pg] = {"synthetic", 1};
    matCase[dL] = d;
    matCase[ek] = eps2;
    w1Case[ok] = omega;
    w1Case[k] = k1;
    w1Case[EE] = e1;
    w2Case[ok] = omega;
    w2Case[k] = k2;
    w2Case[EE] = e2;
    w11Case[ok] = 2*omega;
    w11Case[k] = 2*k1;
    w11Case[EE] = {C11, C21, C31};
    w22Case[ok] = 2*omega;
    w22Case[k] = 2*k2;
    w22Case[EE] = {C12, C22, C32};
    wmixCase[ok] = 2*omega;
    wmixCase[k] = k1 + k2;
    wmixCase[EE] = {C13, C23, C33};
  ];
  solveInhom[matCase, w1Case, w2Case, w11Case, w22Case, wmixCase];
  <|
    "id" -> StringTemplate["solve_inhom_``"][IntegerString[idx, 10, 2]],
    "inputs" -> <|
      "omega" -> jsonValue[omega],
      "epsilon_2omega_lab" -> jsonValue[eps2],
      "d_voigt_lab" -> jsonValue[d],
      "e1" -> jsonValue[e1],
      "e2" -> jsonValue[e2],
      "k1" -> jsonValue[k1],
      "k2" -> jsonValue[k2]
    |>,
    "outputs" -> <|
      "ee" -> jsonValue[w11Case[EE]],
      "oo" -> jsonValue[w22Case[EE]],
      "eo" -> jsonValue[wmixCase[EE]]
    |>
  |>
];

cases = Table[makeCase[idx], {idx, 1, 20}];

Export[
  SHAARPPaths`Ref["solve_inhom_reference_v1.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveInhom reference values",
    "status" -> "mathematica_reference_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"solveInhom" -> Length[DownValues[solveInhom]], "computePNL" -> Length[DownValues[computePNL]]|>,
    "selectedKeys" -> <|
      "omega" -> ToString[omegaKey, InputForm],
      "epsilon2omegaL" -> ToString[eps2OmegaLKey, InputForm],
      "epsilon2omegaLCharacterCodes" -> ToCharacterCode[SymbolName[eps2OmegaLKey], "Unicode"]
    |>,
    "cases" -> cases
  |>,
  "JSON"
];
Quit[]
