# Break-over diode (BOD) - Simscape model and technical note

Rev. 00, 2026-09-22

- `+dtt/+crowbar/bod.ssc` - Simscape component `dtt.crowbar.bod`: functional BOD model with
  switching dynamics (breakover, dv/dt trigger, turn-on time, holding-current turn-off,
  tq surrogate, reverse breakdown, optional thermal port). Defaults: IXBOD2-11 (1100 V).
  Not yet compiled in MATLAB: run `ssc_build dtt` and check.
- `doc/bod_model.tex`, `doc/bod_model.pdf` - technical note (English): device, application
  circuits and design rules, model equations, parameters, verification, listing, bibliography.
  Compile with `pdflatex --shell-escape bod_model.tex` (twice); figures in `doc/fig/`.
- `py/bod_sim.py` - Python replica of the model equations used for the verification figures
  (numpy, scipy, matplotlib): `python bod_sim.py` regenerates `doc/fig/*.pdf`.
