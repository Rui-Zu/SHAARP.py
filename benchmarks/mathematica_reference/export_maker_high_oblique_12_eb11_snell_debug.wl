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
held = ToExpression[cells, StandardForm, HoldComplete];
Scan[ReleaseHold, held];

inputPath = SHAARPPaths`Ref["maker_mathematica_inputs_v1.json"];
outputPath = SHAARPPaths`Ref["maker_high_oblique_12_eb11_snell_debug_v1.json"];
inputPayload = Import[inputPath, "RawJSON"];

c0 = 1;
\[Mu]0 = 1;
\[CurlyEpsilon]0 = 1;
debug = False;
flagLinearSolve = False;
flagNSolve = True;

jsonValue[x_List] := jsonValue /@ x;
jsonValue[x_Association] := AssociationMap[jsonValue, x];
jsonValue[x_?NumericQ] := <|"real" -> N[Re[x], 17], "imag" -> N[Im[x], 17]|>;
jsonValue[x_] := ToString[x, InputForm];

fromJsonComplex[x_Association] := N[x["real"] + I*x["imag"], 17];
fromJsonComplex[x_List] := fromJsonComplex /@ x;
fromJsonComplex[x_?NumericQ] := N[x, 17];
safeReal[x_] := Re[N[x, 17]];

makeMaterial[layer_Association] := Module[
  {orientationRows, qc, epsW, eps2W, d, hValue, pgValue, mat},
  orientationRows = safeReal /@ fromJsonComplex[layer["orientation_rows"]];
  qc = safeReal /@ fromJsonComplex[layer["qc"]];
  epsW = fromJsonComplex[layer["epsilon_omega_crystal"]];
  eps2W = fromJsonComplex[layer["epsilon_2omega_crystal"]];
  d = fromJsonComplex[layer["d_voigt_crystal"]];
  hValue = If[layer["thickness_um"] === Null, 0., N[layer["thickness_um"], 17]];
  pgValue = {layer["point_group"], ConstantArray[0, {3, 6}], layer["nonlinear_flag"]};
  mat = setMater[{
      layer["material_name"],
      {2, {{0, 0, 1}, {0, 1, 0}}, orientationRows},
      pgValue,
      layer["latcon"],
      epsW,
      eps2W,
      d,
      hValue,
      qc
    }];
  extMater[mat];
  mat
];

SetAttributes[snellDebugSnapshot, HoldAll];
snellDebugSnapshot[mat_, wIn_, wOut1_, wOut2_, reflect_: False] := Module[
  {\[Theta]t, n0, ux, uy, uz, L, R, V, W, \[CapitalLambda], C, \[Theta]i,
   mFast, mSlow, QP0, nfastExpr, nslowExpr, nfast, nslow, \[Theta]tfast,
   \[Theta]tslow, kfast, kslow, sol, \[Eta]Fast, rawFast, transformedFast,
   \[Eta]Slow, rawSlow, transformedSlow, selectedFast, selectedSlow,
   selectedFastEta, selectedSlowEta, \[Theta]guess, \[CurlyEpsilon]P,
   \[CurlyEpsilon]L, resolu = 10^(-8), crit0 = 10^(-6), generalizedFastEta,
   generalizedFastVectors, generalizedSlowEta, generalizedSlowVectors},
  \[Theta]i = wIn[\[Theta]];
  n0 = (c0/wIn[\[Omega]])*wIn[k0];
  If[Abs[wOut1[\[Omega]] - \[Omega]0] <= 10^(-3),
    \[CurlyEpsilon]P = mat[\[CurlyEpsilon]\[Omega]P];
    \[CurlyEpsilon]L = mat[\[CurlyEpsilon]\[Omega]L];
    QP0 = mat[QP\[Omega]]
  ];
  If[Abs[wOut1[\[Omega]] - 2*\[Omega]0] <= 10^(-3),
    \[CurlyEpsilon]P = mat[\[CurlyEpsilon]2\[Omega]P];
    \[CurlyEpsilon]L = mat[\[CurlyEpsilon]2\[Omega]L];
    QP0 = mat[QP2\[Omega]]
  ];
  ux = Sin[\[Theta]t]; uy = 0.; uz = Cos[\[Theta]t];
  L = {{uy^2 + uz^2, (-ux)*uy, (-ux)*uz}, {(-ux)*uy, ux^2 + uz^2, (-uy)*uz}, {(-uz)*ux, (-uz)*uy, ux^2 + uy^2}};
  R = MatrixPower[\[CurlyEpsilon]P, 1/2] . Transpose[QP0];
  C = Inverse[R];
  \[CapitalLambda] = Transpose[C] . L . C;
  V = (1/2)*(\[CapitalLambda][[1, 1]] + \[CapitalLambda][[2, 2]] + \[CapitalLambda][[3, 3]]);
  W = \[CapitalLambda][[1, 1]]*\[CapitalLambda][[2, 2]] + \[CapitalLambda][[1, 1]]*\[CapitalLambda][[3, 3]] + \[CapitalLambda][[2, 2]]*\[CapitalLambda][[3, 3]] - \[CapitalLambda][[3, 1]]*\[CapitalLambda][[1, 3]] - \[CapitalLambda][[1, 2]]*\[CapitalLambda][[2, 1]] - \[CapitalLambda][[2, 3]]*\[CapitalLambda][[3, 2]];
  nfastExpr = Simplify[Sqrt[(V - Sqrt[V^2 - W])/W]];
  nslowExpr = Simplify[Sqrt[(V + Sqrt[V^2 - W])/W]];
  If[Element[Flatten[Chop[\[CurlyEpsilon]P]], Reals],
    If[reflect, \[Theta]guess = Pi - \[Theta]i, \[Theta]guess = \[Theta]i],
    If[reflect, \[Theta]guess = Pi - \[Theta]i + 0.1*I, \[Theta]guess = \[Theta]i + 0.1*I]
  ];
  Quiet[sol = FindRoot[Sin[\[Theta]t]*nfastExpr == n0*Sin[\[Theta]i], {\[Theta]t, \[Theta]guess}]];
  \[Theta]tfast = \[Theta]t /. sol;
  nfast = nfastExpr /. sol;
  Quiet[sol = FindRoot[Sin[\[Theta]t]*nslowExpr == n0*Sin[\[Theta]i], {\[Theta]t, \[Theta]guess}]];
  \[Theta]tslow = \[Theta]t /. sol;
  nslow = nslowExpr /. sol;
  If[Abs[Im[\[Theta]tfast]] <= resolu, \[Theta]tfast = Re[\[Theta]tfast]];
  If[Abs[Im[\[Theta]tslow]] <= resolu, \[Theta]tslow = Re[\[Theta]tslow]];
  kfast = (nfast*(wOut1[\[Omega]]/c0))*{Sin[\[Theta]tfast], 0, Cos[\[Theta]tfast]};
  kslow = (nslow*(wOut2[\[Omega]]/c0))*{Sin[\[Theta]tslow], 0, Cos[\[Theta]tslow]};

  mFast = \[CapitalLambda] /. \[Theta]t -> \[Theta]tfast;
  {\[Eta]Fast, rawFast} = If[HermitianMatrixQ[mFast], Eigensystem[mFast, 2, Method -> "Banded"], Eigensystem[mFast, 2]];
  transformedFast = Normalize /@ (rawFast . Transpose[C]);
  selectedFastEta = Select[Thread[{\[Eta]Fast, transformedFast}], Abs[#1[[1]] - 1/nfast^2] <= resolu &][[1, 1]];
  selectedFast = Select[Thread[{\[Eta]Fast, transformedFast}], Abs[#1[[1]] - 1/nfast^2] <= resolu &][[1, 2]];

  mSlow = \[CapitalLambda] /. \[Theta]t -> \[Theta]tslow;
  {\[Eta]Slow, rawSlow} = If[HermitianMatrixQ[mSlow], Eigensystem[mSlow, 2, Method -> "Banded"], Eigensystem[mSlow, 2]];
  transformedSlow = Normalize /@ (rawSlow . Transpose[C]);
  selectedSlowEta = Select[Thread[{\[Eta]Slow, transformedSlow}], Abs[#1[[1]] - 1/nslow^2] <= resolu &][[1, 1]];
  selectedSlow = Select[Thread[{\[Eta]Slow, transformedSlow}], Abs[#1[[1]] - 1/nslow^2] <= resolu &][[1, 2]];

  {generalizedFastEta, generalizedFastVectors} = Eigensystem[{L, \[CurlyEpsilon]L} /. \[Theta]t -> \[Theta]tfast, 2];
  {generalizedSlowEta, generalizedSlowVectors} = Eigensystem[{L, \[CurlyEpsilon]L} /. \[Theta]t -> \[Theta]tslow, 2];

  wOut1[[Key[k]]] = Chop[kfast, crit0];
  wOut2[[Key[k]]] = Chop[kslow, crit0];
  wOut1[[Key[k0]]] = nfast*(wOut1[\[Omega]]/c0);
  wOut2[[Key[k0]]] = nslow*(wOut2[\[Omega]]/c0);
  wOut1[[Key[\[Theta]]]] = \[Theta]tfast;
  wOut2[[Key[\[Theta]]]] = \[Theta]tslow;
  wOut1[[Key[EE]]] = wOut1[EE]*Chop[selectedFast, crit0];
  wOut2[[Key[EE]]] = wOut2[EE]*Chop[selectedSlow, crit0];

  <|
    "reflect" -> reflect,
    "eigenStandard" -> eigenStandard,
    "theta_i" -> jsonValue[N[\[Theta]i, 17]],
    "theta_guess" -> jsonValue[N[\[Theta]guess, 17]],
    "n0" -> jsonValue[N[n0, 17]],
    "epsilon_P" -> jsonValue[N[\[CurlyEpsilon]P, 17]],
    "epsilon_L" -> jsonValue[N[\[CurlyEpsilon]L, 17]],
    "QP0" -> jsonValue[N[QP0, 17]],
    "R" -> jsonValue[N[R, 17]],
    "C" -> jsonValue[N[C, 17]],
    "theta_fast" -> jsonValue[N[\[Theta]tfast, 17]],
    "theta_slow" -> jsonValue[N[\[Theta]tslow, 17]],
    "nfast" -> jsonValue[N[nfast, 17]],
    "nslow" -> jsonValue[N[nslow, 17]],
    "kfast" -> jsonValue[N[kfast, 17]],
    "kslow" -> jsonValue[N[kslow, 17]],
    "mFast" -> jsonValue[N[mFast, 17]],
    "mSlow" -> jsonValue[N[mSlow, 17]],
    "etaFast" -> jsonValue[N[\[Eta]Fast, 17]],
    "etaSlow" -> jsonValue[N[\[Eta]Slow, 17]],
    "rawFastEigenvectors" -> jsonValue[N[rawFast, 17]],
    "rawSlowEigenvectors" -> jsonValue[N[rawSlow, 17]],
    "transformedFastEigenvectors" -> jsonValue[N[transformedFast, 17]],
    "transformedSlowEigenvectors" -> jsonValue[N[transformedSlow, 17]],
    "selectedFastEta" -> jsonValue[N[selectedFastEta, 17]],
    "selectedSlowEta" -> jsonValue[N[selectedSlowEta, 17]],
    "selectedFastElectric" -> jsonValue[N[Chop[selectedFast, crit0], 17]],
    "selectedSlowElectric" -> jsonValue[N[Chop[selectedSlow, crit0], 17]],
    "generalizedFastEta" -> jsonValue[N[generalizedFastEta, 17]],
    "generalizedSlowEta" -> jsonValue[N[generalizedSlowEta, 17]],
    "generalizedFastVectors" -> jsonValue[N[Normalize /@ generalizedFastVectors, 17]],
    "generalizedSlowVectors" -> jsonValue[N[Normalize /@ generalizedSlowVectors, 17]]
  |>
];

case = SelectFirst[inputPayload["cases"], #["id"] == "maker_fringes_moderate_high_oblique" &];
If[MissingQ[case],
  Print["Missing maker_fringes_moderate_high_oblique in input manifest."];
  Quit[1]
];

mats = makeMaterial /@ case["layers"];
\[Omega]0 = N[case["omega0"], 17];
\[CapitalDelta]\[Delta] = N[case["ellipticity_deg"] Degree, 17];
\[CurlyPhi] = N[case["phi_deg"] Degree, 17];
\[Psi] = N[case["psi_deg"] Degree, 17];
n0 = Sqrt[Mean[Eigenvalues[mats[[1]][\[CurlyEpsilon]\[Omega]C]]]];

thetaDeg = 12.;
wIncLocal = setwInc[\[Omega]0, N[(thetaDeg/180) Pi, 17], \[CapitalDelta]\[Delta], \[CurlyPhi], n0];
extWave @ wIncLocal;

wF1Local = setWave[{\[Omega]0, kF11, k0F11, \[Theta]F11, EF11}];
wF2Local = setWave[{\[Omega]0, kF21, k0F21, \[Theta]F21, EF21}];
solveSnell[mats[[2]], wIncLocal, wF1Local, wF2Local];

wB1Local = setWave[{\[Omega]0, kB11, k0B11, \[Theta]B11, EB11}];
wB2Local = setWave[{\[Omega]0, kB21, k0B21, \[Theta]B21, EB21}];
snapshot = snellDebugSnapshot[mats[[2]], wF1Local, wB1Local, wB2Local, True];

Export[
  outputPath,
  <|
    "source" -> "Mathematica SHAARP.ml setup.nb theta=12 EB11 Snell/eigenvector debug",
    "status" -> "mathematica_maker_high_oblique_12_eb11_snell_debug_exported",
    "wolfram_version" -> $Version,
    "input_manifest" -> inputPath,
    "case_id" -> "maker_fringes_moderate_high_oblique",
    "theta_deg" -> thetaDeg,
    "debug_snapshot" -> snapshot
  |>,
  "JSON"
];
Quit[]
