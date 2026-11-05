# Repositories documentation

Open-source documentation for the public repositories of the project:
[library](https://github.com/pwr-control/library),
[modelization-and-control](https://github.com/pwr-control/modelization-and-control),
[solid-state-transformers](https://github.com/pwr-control/solid-state-transformers),
[grid-forming](https://github.com/pwr-control/grid-forming) and
[advanced-control-engineering](https://github.com/pwr-control/advanced-control-engineering).
The documents of the SST project (`repos_doc/solid_state_transformers`, `repos_doc/cllc_dcx`,
`repos_doc/single_phase_grid_tied_inverter_analysis`) and the note on the grid-forming and
grid-following AFE blocks (`repos_doc/grid_forming_following_afe`) are also kept, with the same
sources, in `library/library/documentation/` of the library repository, so that the Simulink
models can link their PDFs (the AFE note is the Help of the two block masks). The design note of the
model `grid-forming/parallel_of_GFMs` (`repos_doc/parallel_of_gfms`) is not copied: the init of the
model puts that folder on the MATLAB path when this repository sits next to `grid-forming`, and the
model opens the PDF from its annotation "Design note".

Everything is LaTeX. Documents are technical reports and notes on power electronics, electrical
machines, semiconductor devices and control, written to accompany the models of the other
repositories.

## Layout

| Folder | Content |
|---|---|
| [`repos_doc`](repos_doc) | Reports documenting the models and studies of the other repositories: machines, converters, hydraulics, batteries, fuel cell, drives, single-phase inverters, PLLs, solid-state transformers and their CLLC-DCX stage, grid-forming and grid-following AFE blocks, parallel grid-forming modules (power sharing, earthing, common mode), power supplies, IGBT selection |
| [`research`](research) | Shorter technical notes grouped by domain: electrical machines and transformers, electronics, mathematics, physics, power electronics, power semiconductor devices, power systems, signal processing, space vector modulation, varistors, system engineering |
| `standards_analysis` | Notes on standards: IEC 60034-27-4 (insulation tests on rotating machines, Italian) and the nomenclature of line-commutated converter connections (DIN 41761, IEC 60146, ANSI/IEEE) |
| `texbook` | Tooling that assembles the standalone notes into a single book (see its own README) |
| `posters` | An A3 infographic/poster template (dark background, tcolorbox, TikZ) |
| `AAA_template_latex_settings` | The shared preamble and title-page templates used by every document |

`repos_doc` and `research` have their own README with a document-by-document index.

## Template

Every document starts with `\input{../../AAA_template_latex_settings/settings}`. The preamble
provides British babel, A4 with 2.5 cm margins, TikZ/pgfplots, siunitx, chemistry packages,
theorem environments, a `mybox` framed environment, nomenclature and a page style with the
pwr-control logo. The title page is driven by `\DocCategory`, `\DocField`, `\DocTitle`,
`\DocSubtitle`, `\DocAuthor`, `\DocDate`, `\DocAbstract` and three metadata rows (repository,
simulation tool, hardware platform); `frontespizio_v2.tex` is the standard version,
`colored/frontespizio_template.tex` a coloured variant. `pdf_output_settings.tex` explains how to
set a per-project PDF output directory in TeXstudio on Windows.

## Building a document

```
cd repos_doc/<document>
pdflatex <document>.tex      # twice for references and table of contents
```

## Assembling the book

`texbook/` holds `manifest.yaml` (parts and chapters), `tools/merge_tex.py` (merges the source
documents, unifies the preambles, namespaces labels and citations, merges the bibliographies) and
the generated `book.tex`. Sources go in `texbook/src/`; see `texbook/README.md`.
