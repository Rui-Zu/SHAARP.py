# My Materials — saving your own materials

Both tabs end with a collapsible **My Materials** group (it sits where the former *Layer Properties
Preset Values* group used to be). It lets you keep the material you have entered in the crystal
panels under a name of your own, so it can be selected again later — in another session, on either
tab — exactly like a palette entry.

## What is saved

**Save current as new** stores the material currently in the crystal panels: the point group, the
lattice, the orientation, the $\varepsilon(\omega)$ and $\varepsilon(2\omega)$ grids, the $d$ grid
and the refractive-index spins, together with the **current wavelength**. Layer thickness is not
part of a material (it belongs to the layer), nor is anything from the polarimetry or calculation
groups.

Type the name into the text field (placeholder *"name for the current material"*) and click
**Save current as new**. The status bar confirms `Saved material '<name>' (λ = … µm)` and the new
entry is selected immediately. The group's tooltip (hover over it or any of its buttons)
summarizes these rules.

## Where it lives

Your materials are written to `~/.shaarp/user_materials.json` — one small JSON file in the
per-user SHAARP folder, separate from the program. `~` means your home folder: on Windows
`C:\Users\<your name>\.shaarp\`, on macOS `/Users/<your name>/.shaarp/`. Living outside the app
folder, it survives updating or re-downloading SHAARP.py. It is a user store only: the built-in case
studies and palette films are never modified, and a name that matches a built-in case is refused
(`'GaAs (111)' is a built-in material or a reserved name; choose another`).

## Where a saved material appears

- **SHAARP.si tab** — at the end of the *Case Study and Examples* dropdown, under a header row
  **— My Materials —** (the header itself cannot be selected), after *Polar Metals*.
- **SHAARP.ml tab** — in the N-layer stack editor's **layer material** list, under the same
  **— My Materials —** header, just before *Custom (fields)*. It is **not** listed in the
  *Single film in air* section of the system dropdown; to use it as a film, open the
  *N-layer stack (editor)* and assign it to a layer.

## One material, one wavelength

A saved material is **single-wavelength**: its tensors were entered at the wavelength in effect when
you saved it. Selecting it therefore sets the wavelength spin to that value and the row label
reads *"wavelength (µm) — set by the case (single-λ data)"*, the same rule the built-in
single-λ cases follow (e.g. *TaAs (112)* → 0.8 µm, *GaAs (111) @1064 nm* → 1.064 µm). Editing
the spin while the material is selected snaps it back. To use the same crystal at another
wavelength, select *Custom (use fields)*, enter the tensors for that wavelength, and save it under a
second name.

## Update, rename, delete

The **Update selected**, **Rename…** and **Delete selected** buttons are enabled only while one of
*your* materials is the current selection; with a built-in case or *Custom* selected they are
disabled and the info label reads *"selected: — (pick one of your materials to update / rename /
delete)"*. With one of your materials selected the label shows its name, the time it was saved,
and its wavelength.

- **Update selected** overwrites the selected material with the current panels (and the current
  wavelength) after a confirmation prompt.
- **Rename…** asks for a new name; the built-in-name rule applies here too. On the ML tab, stack
  rows that use the material follow the new name.
- **Delete selected** removes the material after a confirmation prompt. If it was the current
  selection, the dropdown falls back to *Custom (use fields)* / *Custom (fields)* and the status bar
  notes `Material '<name>' was deleted — switched to Custom.`; on the ML tab, stack rows that
  carried it keep their values as custom layers.

## Sharing materials

The store is an ordinary JSON file. To hand your materials to a colleague (or move them to another
machine), copy `~/.shaarp/user_materials.json`; SHAARP.py reads it at start-up and whenever the
My Materials group refreshes. A missing or unreadable file simply means an empty list.

## The old Presets group

The session-scoped **Layer Properties Preset Values** group (numbered preset buttons, *Clear
Presets*, *Show Preset Info*) is superseded by My Materials and is no longer shown. Presets never
persisted across sessions; My Materials does.
