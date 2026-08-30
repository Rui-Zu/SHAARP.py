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
jsonValue[x_] := ToString[x, InputForm];

globalNames = Names["Global`*"];
omegaNameMatches = Select[globalNames, StringLength[#] <= 20 && StringContainsQ[#, "Global`"] &];
omegaKey = \[Omega];
eps2OmegaLKey = ToExpression["Global`" <> FromCharacterCode[{949, 50, 969, 76}, "Unicode"]];

d = {
  {0.3 + 0.2*I, -0.4*I, 0.7, 0.2 - 0.1*I, 0.05*I, -0.6},
  {0.1*I, 0.5 + 0.15*I, -0.25, 0.45*I, 0.8, -0.2 + 0.05*I},
  {-0.15, 0.25*I, 0.35 + 0.2*I, -0.5, 0.65*I, 0.4}
};
eps2 = {
  {2.6 + 0.15*I, 0.08 - 0.03*I, -0.05 + 0.02*I},
  {0.04 + 0.07*I, 3.1 + 0.11*I, 0.06 - 0.01*I},
  {-0.02 + 0.05*I, 0.03 + 0.04*I, 3.7 + 0.09*I}
};
With[{ok = omegaKey, ek = eps2OmegaLKey},
  mat0[pg] = {"synthetic", 1};
  mat0[dL] = d;
  mat0[ek] = eps2;
  w1s[ok] = 1.4;
  w1s[k] = {0.31, 0., 1.12};
  w1s[EE] = {0.8 + 0.1*I, 0.04*I, -0.22};
  w2s[ok] = 1.4;
  w2s[k] = {0.31, 0., 1.37};
  w2s[EE] = {0.03, 0.91 - 0.18*I, 0.11*I};
  w11s[ok] = 2.8;
  w11s[k] = 2*w1s[k];
  w11s[EE] = {C11, C21, C31};
  w22s[ok] = 2.8;
  w22s[k] = 2*w2s[k];
  w22s[EE] = {C12, C22, C32};
  wmixs[ok] = 2.8;
  wmixs[k] = w1s[k] + w2s[k];
  wmixs[EE] = {C13, C23, C33};
];

solveInhomResult = solveInhom[mat0, w1s, w2s, w11s, w22s, wmixs];

Export[
  SHAARPPaths`Ref["solve_inhom_smoke_output.json"],
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb solveInhom smoke",
    "wolfram_version" -> $Version,
    "inputCellCount" -> SHAARPReferenceLoader`inputCellCount,
    "selectedKeys" -> <|
      "omega" -> ToString[omegaKey, InputForm],
      "epsilon2omegaL" -> ToString[eps2OmegaLKey, InputForm],
      "matPgLast" -> ToString[mat0[pg][[-1]], InputForm],
      "w1Omega" -> With[{ok = omegaKey}, ToString[w1s[ok], InputForm]],
      "matEpsHead" -> With[{ek = eps2OmegaLKey}, ToString[Head[mat0[ek]], InputForm]],
      "nameSample" -> Take[globalNames, UpTo[80]]
    |>,
    "functionDownValues" -> <|"solveInhom" -> Length[DownValues[solveInhom]]|>,
    "solveInhomResult" -> ToString[Short[solveInhomResult, 5], InputForm],
    "outputs" -> <|
      "ee" -> jsonValue[w11s[EE]],
      "oo" -> jsonValue[w22s[EE]],
      "eo" -> jsonValue[wmixs[EE]]
    |>
  |>,
  "JSON"
];
Quit[]
