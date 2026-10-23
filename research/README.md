# research — technical notes by domain

Shorter notes than the reports in `repos_doc`, each in its own folder with the `.tex` source, the
PDF and, where present, the companion script or Simscape component (`.m`, `.py`, `.ssc`,
`.mlx`). English unless marked (IT). Several of the `.ssc` files here are the source of the
components shipped in the [library](https://github.com/pwr-control/library) repository.

## `power_semiconductor_devices`

Physics:
- `physics/solid_state` — *Derivation and physical meaning of the Fermi level in semiconductors*.
- `diode/power_diode` — *Il power diode a giunzione PIN* (IT): carriers, Fermi levels, forward
  conduction, reverse recovery, lifetime control, MPS diodes.
- `diode/charge_control_model` — *Dall'equazione di continuità ambipolare al modello a controllo
  di carica* (IT): why the two-state Lauritzen-Ma form is needed for di/dt-dependent recovery.
- `igbt/igbt_fermi_levels`, `thyristor/thyristor_fermi_levels` — band diagrams with Fermi and
  quasi-Fermi levels in the rest, blocking and conduction states (IT).
- `thyristor/thyristor` — *Il tiristore (SCR): principio di funzionamento e fisica del
  dispositivo* (IT): two-transistor model linked to the datasheet parameters.
- `gto_igct` — *GTO and IGCT thyristors: device principles, switching behaviour and gate driver
  design*, with bibliography.

Simscape device models:
- `igbt/igbt_dyn` — *Modelli Simscape per la dinamica di commutazione: diode_rr e igbt_dyn* (IT).
- `power_mosfet/pwr_mosfet_dyn` — *Modello Simscape mos_dyn* (IT), with `mos_dyn.ssc`.
- `igbt/igbt_ssc_model` — *A simplified Simscape IGBT model with single-point switching energies
  and thermal ports*, with `igbt_simple.ssc` and `igbt_simple_2th.ssc`.
- `power_mosfet/pwr_mosfet_simple` — *mos_simple_2th: a simplified power MOSFET loss model*,
  with `mos_simple_2th.ssc`.
- `thyristor/thyristor_ssc_model` — *A Simscape language thyristor model with reverse recovery*
  (Lauritzen-Ma dynamics, latching logic, tq constraint), with `thyristor_rr.ssc`.
- `tvs_clamp` — *Active clamping of high-voltage IGBTs* (earlier copy of the note under
  `power_electronics/active_clamping`), with `tvs_clamp.ssc`.
- `mosfet/mosfet_dyn` — PDF of the `mos_dyn` note only.

Gate drivers and snubbers:
- `thyristor/thyristor_driver` — *Design guidelines for thyristor gate drivers* (EN and IT
  versions) with four driver topologies and a 6.5 kV / 2.6 kA press-pack example;
  `firing_channel.tex` is a circuitikz schematic of one firing channel.
- `snubbers` — three notes: snubber design guidelines for a half-bridge leg with FZ2400R33HE4
  modules; snubbers in paralleled 5SNA 1000G650300 modules; RC snubber for high-power three-phase
  diode bridges (IT, 690 V / 4 kA example).

## `power_electronics`

Converters and control:
- `single_phase_afe_traction_line` — *Convertitore di rete monofase 250 kW per l'immissione di
  energia eolica in rete di trazione 16.7 Hz* (IT) and *Energy-based control of a full-bridge
  active power decoupling circuit for a 350 kW single-phase AFE*.
- `single_phase_inverter` — phase relation between the line quantities at ω and the DC-link
  ripple at 2ω.
- `dead_time` — *Dead time in voltage source converters*: symmetric and asymmetric insertion,
  harmonic effects with emphasis on even harmonics, compensation.
- `mpc_three_phase_ttype_inverter_fpga` — *FCS-MPC for a three-level T-type inverter: FPGA
  implementation on Xilinx Zynq-7000*.
- `buck_boost_converter` — steady-state analysis: inductor and output current versus duty cycle.
- `upqc` — *A flipped unified power quality conditioner for active distribution networks*
  (PCIM-style paper).
- `converter_nomenclature` — *Nomenclature of line-commutated converter connections* (DIN 41761,
  IEC 60146, ANSI/IEEE, AFE/AIC).
- `LRCL_grid_filter_design` — two live scripts sizing an L-RC-L grid filter for the 400 V /
  250 kVA and 690 V / 1600 kVA cases; sub-notes `summary_EN50160` (IT) and `trafo_equivalente`
  (sequence equivalent circuits of a three-phase transformer, IT).
- `lrc_filter_design_guidelines` — *L(RC) filter design guidelines*: grid-side L / LCL / L(RC)L
  filters for active front ends (exact PWM inductor ripple for SPWM, SVM and DPWM, the ripple-inductance
  trade-off of the reactor with its loss optimum versus the switching frequency, capacitor reactive
  power, attenuation against IEEE 519, resonance placement versus the critical frequency $f_s/6$,
  passive damping and its losses, grid-impedance interaction) and LC / L(RC) sine and dv/dt output
  filters for induction and PMSM drives (reflected wave with long cables, resonance window, loaded
  resonance set by the machine inductance, damping, consequences for the current control and for
  sensorless control), with two step-by-step worked examples for a 450 kW, 480 V, 2 kHz drive (AFE
  grid filter; IM and PMSM sine filters), a review of a 690 V / 250 kW / 4 kHz L(RC) design without
  $L_2$ (grid as second inductance) with its aligned version, and a comparison table;
  `lrc_filter_example.py` reproduces the numbers.

Line-commutated rectifiers and protection:
- `six_pulses_thyristor_based_rectifier/acdc_thyristor_sizing_src` — *Sizing of a 400 V / 1 MW
  thyristor AC/DC supply for a 1100 A resistive load at 16.4–200 V*, with `calc.py` generating
  the tables.
- `six_pulses_thyristor_based_rectifier/rectifier_16V4_1100A_src` — *Sizing of a 16.4 V / 1100 A
  six-pulse thyristor rectifier* (transformer, line reactor with 5th/7th/11th traps, DC LC
  filter), with `calc2.py`.
- `six_pulse_controlled_rectifier_design` — *Six-pulse thyristor bridge (B6C) with arm fuses, RC
  snubbers and DC LC filter: sizing formula sheet and design path* (fourteen steps, 440 V /
  1000 A worked example; Rev. 01 adds the RC snubber design criteria), with `b6c_example_calc.py`
  and the switched-circuit check `b6c_snubber_sim.py`.
- `soft_starters_and_snubbers` — *Thyristor-based soft starters for high-power motors*.
- `thyristor_based_crowbar/unidirectional_crowbar` — *Crowbar thyristor for the 3 kV / 3600 A
  static-switch test bench*: device selection and passive self-triggering gate circuit, with
  `gate_network_sim.py`.
- `thyristor_based_crowbar/bidirectional_crowbar` — *Bidirectional self-triggered thyristor
  crowbar*, with `crowbar_bidir.ssc`.
- `dump_resistor_energy_extraction` — dump resistor as alternative to repetitive MOV clamping
  for million-cycle test-bench duty.
- `active_clamping` — *Active clamping of high-voltage IGBTs: theory, design for a 6.5 kV /
  1000 A module and a Simscape TVS model* (with the adiabatic energy-budget section).
- `double_pulse_test` — *Extraction of Eon, Eoff and Err from double pulse test waveforms*
  (FF1200R17IP5), with the MATLAB procedure.
- `igbt_ageing` — *Aging estimation of an IGBT-based drive* (Arrhenius model with current and
  coolant temperature thresholds), EN and IT.

Magnetics and thermal:
- `dab_cllc_inductor_sizing` — series inductance design in DAB versus CLLC.
- `planar_transformer` — *Trasformatore planare: nota tecnica di dimensionamento* (IT), with
  `planar_xfmr_design.m`.
- `cooling_plate` — thermal impedance of a liquid-cooled aluminium plate (IT and EN).
- `cooling_plate_mersen` — the same on a Mersen pin-fin cold plate, with `thermal_cold_plate.m`.
- `shielded_cable_model` — *Lumped RLC formulation for the Simscape implementation of a shielded
  three-phase cable* (120 m, 90 mm²).

## `electrical_machines`

- `pmsm` — *Torque ripple and cogging torque in PMSM* and its propagation through a back-to-back
  drive into interharmonic grid emission.
- `dy11_fault_analysis` — Dy11 secondary voltages under a single-phase sag on the primary.
- `parallel_transformers` — passive current sharing in parallel-connected medium-frequency
  transformers under parametric mismatch.
- `tap_staggering_master_follower` — master-follower on-load tap changer control for parallel
  transformers.
- `transformers/scott` — *The Scott-T transformer*, with `scott_t.ssc`.
- `transformers/tri_monophase_transformer` — *Three-phase to single-phase transformer with two
  star-connected primaries*: magnetic model of the three-limb core (positive- and zero-sequence
  magnetizing inductances), secondary turns distribution on the limbs, no-load ratio and phase,
  line currents reflected by the single-phase load, short-circuit impedance, role of the neutral
  and of the zero-sequence flux, two primaries in parallel or on two sources; with
  `tri_monophase_transformer.ssc` and the Python check script that reproduces the numbers.
- `transformers/transformer_sizing_theory` — single-phase HF high-power transformer sizing
  (800 V / 800 V, 10 kHz, 250 kW example).
- `transformers/induttanza_dispersione` — leakage inductance from nameplate data (IT).
- `polar_diagram` — MATLAB script drawing the polar diagram of a balanced three-phase set.

## `power_systems`

- `even_harmonics` — *Even harmonics in three-phase AFE rectifiers: origins, mechanisms and
  mitigation*.
- `three_phase_power_system_fault_analaysis` — grid voltage during a single phase-to-earth fault,
  phasor analysis in the RST plane for the different neutral earthing cases.
- `dy11_fault_analysis` — the Dy11 note extended with the grid-voltage constraint for AFE
  operation.

## Other domains

- `electronics/opamp_current_feedback` — *Modello Simscape di amplificatore a retroazione di
  corrente (CFA)* (IT), LT1210 parameterization, with `opamp_cfa.ssc`.
- `varistors/varistor_B80K680` — *The EPCOS SIOV-B80K680 block varistor: operating
  characteristics and application sizing*, with `varistor_mov.ssc` and `test_varistor_mov.m`.
- `svm` — *Space vector modulation for a two-level three-phase inverter*: classical SVM versus
  triangular third-harmonic injection.
- `signal_processing/phasor_dft_explained` — three-phase phasor extraction via DFT with MATLAB
  implementation.
- `cable_acustic_model` — electromagnetic-acoustic model of PWM-excited differential forces on a
  120 m single-core cable run.
- `physics/thermal_diffusion` — 1D thermal diffusion (EN and IT) with `thermal_diffusion_bar.m`.
- `physics/comsol` — Anand viscoplastic model of the SAC305 solder layer for COMSOL (IT).
- `physics/fermi_levels` — PDF only.
- `mathematics/graph_theory` — *Teoria dei grafi algebrica*, summary of Bullo's *Lectures on
  Network Systems* (IT).
- `system_engineering/gantt` — pgfgantt template for a proposal-attachment Gantt chart, with its
  own README and Makefile.
- `shielded_cable_model` — PDF of the cable-model note (source under `power_electronics`).

## Notes

`cooling_plate/thermal_impedance.tex` and `cooling_plate_mersen/thermal_impedance.tex` are
identical; `tvs_clamp` and `power_electronics/active_clamping` are two revisions of the same note.
