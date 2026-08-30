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

eigenStandard = True;
debug = False;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
sym[codes_List] := ToExpression["Global`" <> FromCharacterCode[codes, "Unicode"]];

omegaKey = sym[{969}];
thetaKey = sym[{952}];
epsOmegaPKey = sym[{949, 969, 80}];
epsOmegaLKey = sym[{949, 969, 76}];
qpOmegaKey = sym[{81, 80, 969}];

omega0 = 2*Pi/0.8;
c0 = 1;
anglesDeg = {0, 8, 18, 33, 52};

makeEpsilon[seed_] := N[{
  {2.15 + 0.11*seed + I*(0.045 + 0.007*seed), 0.025 + 0.006*seed - 0.014*I, -0.018 + 0.005*seed + 0.019*I},
  {0.025 + 0.006*seed - 0.014*I, 2.65 + 0.09*seed + I*(0.055 + 0.006*seed), 0.034 - 0.004*seed + 0.016*I},
  {-0.018 + 0.005*seed + 0.019*I, 0.034 - 0.004*seed + 0.016*I, 3.25 + 0.13*seed + I*(0.07 + 0.005*seed)}
}];

makeCase[idx_, tensorSeed_, thetaDeg_, reflectFlag_] := Module[
  {epsLab, thetaIn, matCase, wInCase, wFastCase, wSlowCase},
  epsLab = makeEpsilon[tensorSeed];
  thetaIn = thetaDeg*Degree;
  With[{ok = omegaKey, tk = thetaKey, ep = epsOmegaPKey, el = epsOmegaLKey, qk = qpOmegaKey},
    matCase = <|ep -> epsLab, el -> epsLab, qk -> IdentityMatrix[3]|>;
    wInCase = <|ok -> omega0, tk -> thetaIn, k0 -> omega0|>;
    wFastCase = <|ok -> omega0, EE -> 1|>;
    wSlowCase = <|ok -> omega0, EE -> 1|>;
  ];
  solveSnell[matCase, wInCase, wFastCase, wSlowCase, reflectFlag];
  <|
    "id" -> StringTemplate["solve_snell_``"][IntegerString[idx, 10, 2]],
    "inputs" -> <|
      "theta_deg" -> thetaDeg,
      "reflected" -> reflectFlag,
      "incident_index" -> jsonValue[1],
      "omega" -> jsonValue[omega0],
      "epsilon_lab" -> jsonValue[epsLab]
    |>,
    "outputs" -> <|
      "fast_theta" -> jsonValue[wFastCase[thetaKey]],
      "slow_theta" -> jsonValue[wSlowCase[thetaKey]],
      "fast_refractive_index" -> jsonValue[c0*wFastCase[k0]/wFastCase[omegaKey]],
      "slow_refractive_index" -> jsonValue[c0*wSlowCase[k0]/wSlowCase[omegaKey]],
      "fast_wavevector" -> jsonValue[wFastCase[k]],
      "slow_wavevector" -> jsonValue[wSlowCase[k]],
      "fast_electric_direction" -> jsonValue[wFastCase[EE]],
      "slow_electric_direction" -> jsonValue[wSlowCase[EE]]
    |>
  |>
];

caseSpecs = Flatten[
  Table[{tensorSeed, theta, reflectFlag}, {tensorSeed, {1, 2}}, {reflectFlag, {False, True}}, {theta, anglesDeg}],
  2
];
cases = MapIndexed[makeCase[#2[[1]], #[[1]], #[[2]], #[[3]]] &, caseSpecs];

Export[
  SHAARPPaths`Ref["solve_snell_reference_v1.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveSnell reference values",
    "status" -> "mathematica_reference_exported",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"solveSnell" -> Length[DownValues[solveSnell]]|>,
    "selectedKeys" -> <|
      "omega" -> ToCharacterCode[SymbolName[omegaKey], "Unicode"],
      "theta" -> ToCharacterCode[SymbolName[thetaKey], "Unicode"],
      "epsilonOmegaP" -> ToCharacterCode[SymbolName[epsOmegaPKey], "Unicode"],
      "epsilonOmegaL" -> ToCharacterCode[SymbolName[epsOmegaLKey], "Unicode"],
      "QPOmega" -> ToCharacterCode[SymbolName[qpOmegaKey], "Unicode"]
    |>,
    "case_count" -> Length[cases],
    "cases" -> cases
  |>,
  "JSON"
];
Quit[]
