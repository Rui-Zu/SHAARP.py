"""The path-resolution preamble shared by every Mathematica reference exporter.

The exporters under ``benchmarks/mathematica_reference`` regenerate the reference JSON that the
test suite compares against. They read and write files in three places:

``SHAARP_REF_DIR``
    This checkout's ``benchmarks/mathematica_reference``. Optional -- it defaults to the directory
    the script itself lives in.
``SHAARP_ML_DIR``
    A local checkout of ``github.com/bzw133/SHAARP.ml``, which provides ``setup.nb``.
``SHAARP_SI_DIR``
    A local checkout of ``github.com/Rui-Zu/SHAARP``, which provides ``SHAARP_V1.03``.

The two external repositories are not vendored here, so the scripts that need them abort with an
explanatory message rather than failing obscurely. Running the exporters also requires a licensed
Wolfram kernel; the reference JSON they produce ships, so the test suite never needs them.

The preamble is emitted from this one definition -- by the chunk generator and by the exporters
themselves -- and ``tests/test_wolfram_path_preamble.py`` pins every shipped script to it.
"""
from __future__ import annotations

from pathlib import Path

PREAMBLE = '''(* --- Path resolution ---------------------------------------------------------------
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
'''


def ref(name: str) -> str:
    """Wolfram expression for a file in this checkout's reference directory."""
    return f'SHAARPPaths`Ref["{Path(name).name}"]'


def ml(relative: str) -> str:
    """Wolfram expression for a file inside the external SHAARP.ml checkout."""
    return f'SHAARPPaths`ML["{relative}"]'


def si(relative: str) -> str:
    """Wolfram expression for a file inside the external SHAARP (V1.03) checkout."""
    return f'SHAARPPaths`SI["{relative}"]'
