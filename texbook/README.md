# Power Electronics Notes — collected volume

Scaffolding that turns the standalone `.tex` documents we wrote over the past
months into one `book`-class volume with parts, chapters and a single merged
bibliography.

I do not have your `.tex` files, so this repository ships the machinery and the
table of contents rather than the assembled text. Drop the files in `src/` and
one command produces the book.

## Quick start

```bash
cp /path/to/your/*.tex src/          # your source documents
$EDITOR manifest.yaml                # match the file: names to yours
python3 tools/merge_tex.py --src src --out . --manifest manifest.yaml
pdflatex book && pdflatex book && pdflatex book
```

Missing files are skipped and listed at the end of the run, so you can start
with three documents and add the rest later. `--scan` ignores the manifest and
takes everything in `src/` alphabetically.

A working example is in `demo/` — three small source files with deliberate
collisions, plus the compiled result:

```bash
python3 tools/merge_tex.py --src demo --out . --scan
```

## What the script actually does

| Problem when concatenating documents | Handling |
|---|---|
| 25 preambles, overlapping `\usepackage` | union written to `preamble_generated.tex`; conflicting options reported |
| `\label{fig:structure}` in six files | every label namespaced with the chapter slug; `\ref`, `\eqref`, `\cref`, `\autoref`, `\pageref` rewritten to match |
| `\bibitem{lauritzen1991}` in three files | bibliographies merged, duplicates detected by entry text, `\cite` keys remapped, collisions renamed `key-slug` |
| `\newcommand{\Vf}` meaning two things | first definition global; later chapters get a local `\renewcommand` and a restore at chapter end |
| `\title`/`\maketitle`/`\tableofcontents` | dropped; title becomes the chapter title |
| `\begin{abstract}` in an article | rewritten to `chapabstract`, a chapter-opening italic summary |
| a file that already uses `\chapter` | internal sectioning demoted one level |
| `\appendix` inside a file | would restart numbering for the whole book — replaced by a plain section and flagged |

Nothing in `src/` is modified. `book.tex`, `chapters/*`, `bibliography.tex` and
`preamble_generated.tex` are regenerated on every run — put hand edits in
`preamble.tex` and `frontmatter.tex`, which the script never touches.

## Preamble constraints

`preamble.tex` deliberately avoids `babel` (no Italian data files on your
install), `lmodern`, `biblatex` and `mhchem`, matching what we hit while
building `power_diode.tex`. `microtype` is loaded with `expansion=false`,
otherwise it fails without scalable fonts. The bibliography is a plain
`thebibliography`.

## Contents

The parts and chapter order live in `manifest.yaml`. Current plan:

**I — Power semiconductor devices**
PIN power diode · Fermi levels in a thyristor · Fermi levels in an IGBT ·
GTO and IGCT devices

**II — Device models for circuit simulation**
IGBT loss model · IGBT and diode switching dynamics · MOSFET switching
dynamics · MOSFET loss model · thyristor with Lauritzen-Ma recovery ·
CFA operational amplifier · Scott-T transformer

**III — Gate drivers and firing circuits**
Thyristor gate driver guidelines · the O.C.E.M. E637 firing card

**IV — Converters and applications**
Cycloconverters · thyristor soft starters · snubber design for a 2400 A leg ·
single-phase AFE for a 16.7 Hz traction grid · full-bridge APD energy control ·
DC-link 2ω ripple phase · 25 V / 12 kA PSFB supply

**V — Measurement, standards and system analysis**
Eon/Eoff/Err from double pulse tests · EN 50160 · three-phase transformer
sequence circuits · PMSM torque ripple and cogging · IGBT drive aging model

### Version and language rules applied

* **Thyristor gate driver** existed as two documents with the same scope —
  only the later one is in the manifest.
* No document exists in both an English and an Italian version, so the
  "prefer English" rule never had to discard anything. What remains is a set
  of chapters that were only ever written in Italian; they are marked `[IT]`
  in `manifest.yaml`.
* The Simscape model chapters (part II) document components that also live in
  your `.ssc` library; the model files themselves are not included here.

## Known gaps

* Figures: `\graphicspath` covers `figs/`, `chapters/`, `src/`. If your figures
  sit next to each source document, copy them into `figs/` or add the folders.
* Notation is not harmonised across chapters — the preface says so explicitly
  rather than pretending otherwise. Unifying symbols means rereading every
  derivation; worth doing per part, not globally in one pass.
* Chapters that were never written as English documents stay Italian until
  translated.
