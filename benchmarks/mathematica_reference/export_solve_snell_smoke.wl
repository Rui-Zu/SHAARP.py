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
jsonValue[x_] := ToString[x, InputForm];
sym[codes_List] := ToExpression["Global`" <> FromCharacterCode[codes, "Unicode"]];

omegaKey = sym[{969}];
thetaKey = sym[{952}];
epsOmegaPKey = sym[{949, 969, 80}];
epsOmegaLKey = sym[{949, 969, 76}];
qpOmegaKey = sym[{81, 80, 969}];

epsLab = {
  {2.4 + 0.08*I, 0.03 - 0.01*I, 0.02 + 0.015*I},
  {0.03 - 0.01*I, 2.9 + 0.11*I, -0.025 + 0.02*I},
  {0.02 + 0.015*I, -0.025 + 0.02*I, 3.5 + 0.16*I}
};
thetaIn = 24*Degree;
omega0 = 2*Pi/0.8;
c0 = 1;

With[{ok = omegaKey, tk = thetaKey, ep = epsOmegaPKey, el = epsOmegaLKey, qk = qpOmegaKey},
  matSnell = <|ep -> epsLab, el -> epsLab, qk -> IdentityMatrix[3]|>;
  wInSnell = <|ok -> omega0, tk -> thetaIn, k0 -> omega0|>;
  wFast = <|ok -> omega0, EE -> 1|>;
  wSlow = <|ok -> omega0, EE -> 1|>;
];

solveSnellResult = solveSnell[matSnell, wInSnell, wFast, wSlow];

Export[
  SHAARPPaths`Ref["solve_snell_smoke_output.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveSnell smoke",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"solveSnell" -> Length[DownValues[solveSnell]]|>,
    "solveSnellResult" -> ToString[Short[solveSnellResult, 5], InputForm],
    "selectedKeys" -> <|
      "omega" -> ToCharacterCode[SymbolName[omegaKey], "Unicode"],
      "theta" -> ToCharacterCode[SymbolName[thetaKey], "Unicode"],
      "epsilonOmegaP" -> ToCharacterCode[SymbolName[epsOmegaPKey], "Unicode"],
      "epsilonOmegaL" -> ToCharacterCode[SymbolName[epsOmegaLKey], "Unicode"],
      "QPOmega" -> ToCharacterCode[SymbolName[qpOmegaKey], "Unicode"]
    |>,
    "inputs" -> <|
      "epsilon_lab" -> jsonValue[epsLab],
      "theta_incident_rad" -> jsonValue[thetaIn],
      "incident_index" -> jsonValue[1]
    |>,
    "outputs" -> <|
      "fast_theta" -> jsonValue[wFast[thetaKey]],
      "slow_theta" -> jsonValue[wSlow[thetaKey]],
      "fast_refractive_index" -> jsonValue[c0*wFast[k0]/wFast[omegaKey]],
      "slow_refractive_index" -> jsonValue[c0*wSlow[k0]/wSlow[omegaKey]],
      "fast_wavevector" -> jsonValue[wFast[k]],
      "slow_wavevector" -> jsonValue[wSlow[k]],
      "fast_electric_direction" -> jsonValue[wFast[EE]],
      "slow_electric_direction" -> jsonValue[wSlow[EE]]
    |>
  |>,
  "JSON"
];
Quit[]
