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
mu0 = 1;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
jsonValue[x_] := ToString[x, InputForm];
sym[codes_List] := ToExpression["Global`" <> FromCharacterCode[codes, "Unicode"]];
omegaKey = sym[{969}];

wave[n_, theta_, pol_, amp_, zsign_] := Module[{dir, e},
  dir = {Sin[theta], 0, zsign*Cos[theta]};
  e = If[pol == "s", {0, 1, 0}, {zsign*Cos[theta], 0, -Sin[theta]}];
  <|omegaKey -> 1, k -> n*dir, EE -> amp*e|>
];

n1 = 1.;
n2 = 1.7;
thetaI = 35*Degree;
thetaT = ArcSin[(n1/n2)*Sin[thetaI]];
incidentWaves = {wave[n1, thetaI, "s", 1, 1]};
reflectedWaves = {
  wave[n1, thetaI, "s", rS, -1],
  wave[n1, thetaI, "p", rP, -1]
};
transmittedWaves = {
  wave[n2, thetaT, "s", tS, 1],
  wave[n2, thetaT, "p", tP, 1]
};
unknowns = {rS, rP, tS, tP};

originalCall = solveFresnel[incidentWaves, reflectedWaves, transmittedWaves, unknowns];

getH[wav_] := Cross[Key[k][wav], Key[EE][wav]]/(Key[omegaKey][wav]*mu0);
emTop = Total[(Key[EE][#] &) /@ Flatten[{incidentWaves, reflectedWaves}]];
emBot = Total[(Key[EE][#] &) /@ Flatten[transmittedWaves]];
hmTop = Total[(getH[#] &) /@ Flatten[{incidentWaves, reflectedWaves}]];
hmBot = Total[(getH[#] &) /@ Flatten[transmittedWaves]];
equations = Flatten[Thread /@ {emTop[[1 ;; 2]] == emBot[[1 ;; 2]], hmTop[[1 ;; 2]] == hmBot[[1 ;; 2]]}];
sol = First[NSolve[equations, unknowns]];
wRSolved = reflectedWaves /. sol;
wTSolved = transmittedWaves /. sol;

Export[
  SHAARPPaths`Ref["solve_fresnel_smoke_output.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveFresnel smoke",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "functionDownValues" -> <|"solveFresnel" -> Length[DownValues[solveFresnel]]|>,
    "originalCallReturn" -> ToString[Short[originalCall, 5], InputForm],
    "originalWRStillSymbolic" -> Not[FreeQ[reflectedWaves, rS | rP]],
    "solution" -> jsonValue[unknowns /. sol],
    "reflected_electric" -> jsonValue[Key[EE] /@ wRSolved],
    "transmitted_electric" -> jsonValue[Key[EE] /@ wTSolved]
  |>,
  "JSON"
];
Quit[]
