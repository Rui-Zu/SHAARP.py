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

flagLinearSolve = False;
flagNSolve = True;
mu0 = 1;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
sym[codes_List] := ToExpression["Global`" <> FromCharacterCode[codes, "Unicode"]];
omegaKey = sym[{969}];

wave[n_, theta_, pol_, amp_, zsign_] := Module[{dir, e},
  dir = {Sin[theta], 0, zsign*Cos[theta]};
  e = If[pol == "s", {0, 1, 0}, {zsign*Cos[theta], 0, -Sin[theta]}];
  <|omegaKey -> 1, k -> n*dir, EE -> amp*e|>
];

solveFresnelEquations[incidentWaves_, reflectedWaves_, transmittedWaves_, unknowns_] := Module[
  {getH, emTop, emBot, hmTop, hmBot, equations, sol},
  getH[wav_] := Cross[Key[k][wav], Key[EE][wav]]/(Key[omegaKey][wav]*mu0);
  emTop = Total[(Key[EE][#] &) /@ Flatten[{incidentWaves, reflectedWaves}]];
  emBot = Total[(Key[EE][#] &) /@ Flatten[transmittedWaves]];
  hmTop = Total[(getH[#] &) /@ Flatten[{incidentWaves, reflectedWaves}]];
  hmBot = Total[(getH[#] &) /@ Flatten[transmittedWaves]];
  equations = Flatten[Thread /@ {emTop[[1 ;; 2]] == emBot[[1 ;; 2]], hmTop[[1 ;; 2]] == hmBot[[1 ;; 2]]}];
  sol = First[NSolve[equations, unknowns]];
  <|
    "solution" -> (unknowns /. sol),
    "reflected_electric" -> (Key[EE] /@ (reflectedWaves /. sol)),
    "transmitted_electric" -> (Key[EE] /@ (transmittedWaves /. sol))
  |>
];

anglesDeg = {0, 8, 18, 33, 52};
media = {1.7 + 0.08*I, 2.15 + 0.14*I};
polarizations = {"s", "p"};

makeCase[idx_, n2_, thetaDeg_, pol_] := Module[
  {n1, thetaI, thetaT, incidentWaves, reflectedWaves, transmittedWaves, unknowns, result},
  Clear[rS, rP, tS, tP];
  n1 = 1.;
  thetaI = thetaDeg*Degree;
  thetaT = ArcSin[(n1/n2)*Sin[thetaI]];
  incidentWaves = {wave[n1, thetaI, pol, 1, 1]};
  reflectedWaves = {
    wave[n1, thetaI, "s", rS, -1],
    wave[n1, thetaI, "p", rP, -1]
  };
  transmittedWaves = {
    wave[n2, thetaT, "s", tS, 1],
    wave[n2, thetaT, "p", tP, 1]
  };
  unknowns = {rS, rP, tS, tP};
  result = solveFresnelEquations[incidentWaves, reflectedWaves, transmittedWaves, unknowns];
  <|
    "id" -> StringTemplate["solve_fresnel_``"][IntegerString[idx, 10, 2]],
    "inputs" -> <|
      "n1" -> jsonValue[n1],
      "n2" -> jsonValue[n2],
      "theta_deg" -> thetaDeg,
      "theta_transmitted_rad" -> jsonValue[thetaT],
      "incident_polarization" -> pol
    |>,
    "outputs" -> <|
      "coefficients" -> jsonValue[result["solution"]],
      "reflected_electric" -> jsonValue[result["reflected_electric"]],
      "transmitted_electric" -> jsonValue[result["transmitted_electric"]]
    |>
  |>
];

caseSpecs = Flatten[Table[{n2, theta, pol}, {n2, media}, {theta, anglesDeg}, {pol, polarizations}], 2];
cases = MapIndexed[makeCase[#2[[1]], #[[1]], #[[2]], #[[3]]] &, caseSpecs];

Export[
  SHAARPPaths`Ref["solve_fresnel_reference_v1.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveFresnel tangential-equation reference values",
    "status" -> "mathematica_reference_exported",
    "note" -> "The original solveFresnel DownValue substitutes local wR/wT but does not expose them as a return in batch export; this file solves the exact tangential E/H equations from that DownValue with the same unknown order.",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"solveFresnel" -> Length[DownValues[solveFresnel]]|>,
    "selectedKeys" -> <|"omega" -> ToCharacterCode[SymbolName[omegaKey], "Unicode"]|>,
    "unknown_order" -> {"rS", "rP", "tS", "tP"},
    "case_count" -> Length[cases],
    "cases" -> cases
  |>,
  "JSON"
];
Quit[]
