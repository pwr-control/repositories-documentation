# repos_doc — reports on the models of the other repositories

One folder per report. Each holds the `.tex` source (sometimes with included chapters or a
local `settings.tex`) and, where compiled, the PDF. All are in English unless noted; author
Davide Bagnara. The build is `pdflatex <file>.tex` from inside the folder, since the preamble
is loaded as `../../AAA_template_latex_settings/settings`.

## Electrical machines and drives

- `psm_model_observer_ctrl` — *Permanent Magnet Synchronous Machines: model derivation, control
  and Simscape implementation* (May 2021). PMSM in machine variables, αβ and dq; FOC, MTPA,
  robust field weakening, per-unit normalization; parameter identification from no-load,
  short-circuit and load tests; back-EMF observer and parameter deviation analysis.
- `induction_motor_model_and_control` — *Induction Motor: Modelization and Parameters
  Estimation*. Symmetrical machine, steady-state equivalent circuit, parameter calculation from
  factory tests, reference-frame theory, αβ dynamic model with rotor flux observer, online rotor
  resistance estimation.
- `psm_mpc_ekf` — MPC of a PMSM with an extended Kalman filter observer: control design, EKF for
  the nonlinear machine, space vector and predictive modulators, current reference generator,
  simulation results. (The `\title` still reads "Application to IM…", copied from the next one.)
- `im_mpc_ekf` — *Extended Kalman Filter: Application to IM with T-type MPC Three Phase
  Inverter*. Induction motor model derivation and EKF observer.
- `psm_nonlinear_observer` — *Equazioni dell'osservatore adattativo* (Italian). Equations of an
  adaptive observer for a PMSM in the stator dq frame.
- `inverter_for_drone` — *Flying Basket Inverter Design: model, control strategy and hardware
  dimensioning*. PMSM drive for a drone: parameter identification, DC-link, MOSFET and heatsink
  sizing, electro-thermal simulation, control architectures, firmware, user manual.
- `igbt_selection_and_procedure` — *IGBT for EDrive: selection procedure*. Feasibility of the
  Infineon FF1200R17IP5 as replacement of the DP650B1700, FF650R17IE4 and FF650R17IE4D_B2 in a
  690 V drive: theoretical losses plus double pulse tests with the DB1.00 and DB1.51 driver boards.
- `semiconductor_devices_igbt_1` — *IGBTs performance analysis: application to electrical
  drives* — the same study as a report plus a Beamer presentation (*DPT results, 690 V
  electrical drive*).
- `long_cable_effects_in_electrical_drives` — *Long Power Cable Effects in electrical drives*
  (July 2026). Acoustic emission of a 120 m shielded inverter-motor cable under PWM: on-site
  measurement conclusions, high-frequency cable model (distributed parameters, cascaded π
  sections, Simscape), sound-emission model from Ampère forces to acoustic radiation,
  mitigation, vibration-induced fatigue.
- `modularity` — *Modularity in Power Electronics: impact on grid power quality* (October 2023).
  Two to eight paralleled AFE + inverter + PMSM drives on an MV grid: harmonic model of parallel
  AFEs, THD versus operating point, PWM carriers and PCC impedance, four case scenarios.
- `control-of-string-actuated` — *Control of String-Actuated: following the work of Ruth
  Curtain* (September 2024). Two-pulley string-actuated system: string displacement and torsional
  models, coupled oscillators, control design, one- and two-mode study cases.
- `flexible_shaft` — *Flexible Shaft: model derivation and Simscape implementation* (October
  2022). Two-mass torsional model, resonance and anti-resonance, extension to a generic gear.
- `moving_coil` — *Moving Coil: modelization and control system design* (October 2022).
  Actuator model in moving and stationary frames, control with and without load estimator.

## Grid-tied converters and power supplies

- `single_phase_grid_tied_inverter_analysis` — *Single-Phase Grid-Tied T-Type Inverter: model,
  control design and discrete implementation* (Rev. 01, October 2026; Rev. 00 of June 2020
  compared three control strategies). The controller of the library block
  `afe_single_phase_ttype` (`afe_sp_ttype_ctrl_pwm`), the AC module of the SST bench: LCL plant
  with the line inductance, first harmonic tracker and vector PI on the fundamental of the
  single-phase current with the exact stationary-frame equivalent (a PR controller shaped by the
  tracker: gain stabilization of the 1.4 kHz resonance, DC mode and the condition
  k_i/k_p < 190 rad/s), single-phase PLL as SOGI-QSG, DC-link voltage loop, three-level
  modulator and the discrete implementation. Companion Python script reproduces every number
  and figure.
- `single_phase_inverter` — *Single Phase Inverter: analysis of architecture and control
  strategies* (June 2026), same body updated.
- `three_phase_grid_tied_inverter_analysis` — despite the folder name, another copy of the
  single-phase inverter study.
- `single_phase_afe_power_decoupling` — *Active power decoupling of a single-phase AFE: the ripple
  port and its control in the model `single_phase_afe`* (Rev. 00, October 2026). Why the power of a
  single-phase converter pulsates at 2f and the energy that moves (P/ω); the ripple port of the
  model `single_phase_afe` of `modelization-and-control` (second full bridge, two inductors, buffer
  capacitor at 2000 V, boost type) and the sizing of its parts; the control step by step:
  double-integrator observer that isolates the 100 Hz ripple of the DC-link voltage, phase-shift
  filter that builds its quadrature, rotation at 2θ, vector PI, and why the output is applied
  through the β axis (it compensates the 90° of the DC-link capacitor); buffer voltage and current
  loops; averaged simulation with the discrete loops (ripple 16 % → 0.8 %, third harmonic 13 % →
  0.5 %), check of the numbers and margins of the model, open points. Companion Python script and
  PDF figures in the folder; same source in `library/library/documentation/`.
- `single_phase_lock_loop` — *Single Phase Lock Loop*. SOGI-QSG based PLL and moving-average
  filter implemented on dSPACE SCALEXIO with C-Caller and DFT.
- `three_phase_inductors` — *Three Phase Inductors: mathematical model and construction*. Four-
  versus three-limb reactors (differential and common mode), saturation effects.
- `isop_architecture` — *Four-Winding Transformer: circuit model and governing equations*.
  Flux-linkage matrix, T-equivalent, MMF balance and port relations of the transformer used in
  the input-series output-parallel rail supply.
- `solid_state_transformers` — *Solid State Transformers: architecture, modelization and
  control* (Rev. 01, October 2026). DAB, three-phase DAB and CLLC stages with NPC and T-type
  inverters, control loops, HF transformer design, simulation comparison (the 2025 study), and
  the two-module bench the repository keeps today: CLLC-DCX 800 V / 800 V, 50 kW, 50 kHz and
  T-type single-phase inverter per module, inverters in series on the line, the two models
  (compact library blocks, discrete blocks with losses), `param` structure, timing and
  Backward Euler settings. Figures are referenced without extension (EPS here, PDF in the copy
  kept in `library/library/documentation/`).
- `cllc_dcx` — *CLLC-DCX isolated DC/DC stage: 800 V / 800 V, 50 kW, 50 kHz, sizing, simulation
  model and library block* (Rev. 00, October 2026). The isolated stage of the SST module as an
  open-loop DC transformer: FHA gain, why the stage is not regulated (tank reactance, power
  against phase), resonant tank and magnetizing inductance for ZVS, transformer on stacked
  UU93 ferrite sets, SiC modules and synchronous rectifier drive, DC links, loss budget; the
  35 kHz variant 790 V → 700 V and its origin; the Simscape bench of `solid-state-transformers`
  (modes, switch model, Backward Euler damping against the solver step); the library block
  `cllc_dcx` (ports, measurement and gate vectors, control chain, control log, parameter
  structure). Analytic figures from `py/cllc_dcx_figures.py` (pgfplots data in `figures/`).
  Same source in `library/library/documentation/cllc_dcx/`.
- `grid_forming_following_afe` — *Grid-forming and grid-following AFE blocks: two-level bridge
  with LCL filter, description and user guide* (Rev. 00, October 2026). The library blocks
  `grid_forming_dqctrl` and `grid_following_dqctrl` (one Simscape component for the power
  stage, one MATLAB Function for the control and the PWM): quick-start user guide, power stage
  (two-level bridge, DC link, LCL filter, switch model, measurement and gate vectors), the
  control parts common to both blocks (local clock and carrier, acquisitions, Clarke and Park,
  first harmonic tracker and sequences, vector PI, SVPWM and dead time), the grid-forming chain
  (phase generator, voltage and current loops, master/slave `sync` ports, negative-sequence
  compensator), the grid-following chain (PLL, DC-link loop, zero-sequence loop), ports and
  control logs, the `p` parameter structure with the example values, what the blocks do not
  model. All figures are TikZ. Same source in
  `library/library/documentation/grid_forming_following_afe/`; that PDF is the Help of the two
  block masks.
- `afe_uvw_ctrl` — *AFE control in the uvw frame: the five MATLAB Function blocks of
  `afe_uvw_ctrl`* (Rev. 01, November 2026). The controller of the model `afe_uvw_three_phase` as
  written in the library package `masked_models/afe_uvw_ctrl` (library PR #35): Clarke and
  sequence frames with the fractional-window moving average, grid voltage sequences, fault
  ride-through of VDE-AR-N 4110 (voltage-rate observers, state machine, power limitation,
  pre-fault averages and reactive current references), the three clamped PIs of the reactive
  current loops, the per-phase resonant PI current control in detail (continuous design; the
  forward-backward Euler resonator of the block since library rev. 03, poles inside the unit
  circle, δ = 0.005; the forward Euler resonator of the model and its pole radius: poles outside
  the unit circle for δ < ω ts/2, stable closed loop with the plant anyway; the loop with the
  three resonators), vector limiter, output delay and zero-sequence loop, the new passive
  islanding detector, the parameter structure and its mapping to the model init, the checks
  against the original diagrams (Python twin, Octave loop, MATLAB bench), differences and open
  points. Rev. 01: the forward-backward resonator proposed in rev. 00 is now the one of the block.
  Companion Python script `afe_uvw_ctrl_figures.py` (resonant PI figures and numbers, PDF
  figures in `figures/`).
- `three_phase_dab` — *Three-phase dual active bridge (DAB3): 1200 V / 1200 V, 250 kW, 10 kHz,
  sizing, control, simulation model and runs* (Rev. 00, October 2026). The isolated stage of
  `advanced-dcdc-converters/analysis_three_phase_dab` between two batteries: six-step bridges,
  three single-phase transformers in star-star, power against phase shift and harmonic model,
  comparison with the single-phase DAB, series inductance, transformer and choice of the
  frequency, SiC modules and ZVS map, DC links, loss budget; current loop with feed-forward and
  its sample rate; DC bias of the transformers (origin, gap and ZVS damping, flux-centred start,
  updates of the phase shift, measurement with slow sampling, DC regulators); the Simscape bench
  (`param` structure, changes from the original model, Backward Euler damping, phase
  quantization) and runs 01 to 04 against the harmonic model. Analytic figures from
  `py/three_phase_dab_figures.py`, figures of the runs from `py/three_phase_dab_runs.py`
  (pgfplots data in `figures/`).
- `ps_25V_12kA` — *Power supply for electrolysis* (July 2026). 300 kW, 25 V / 12 kA from a 480 V
  grid: AFE with LCL filter, phase-shift full bridge with centre-tapped HF transformers, diode
  rectifier; design and simulation results.
- `unified_power_quality_conditioner` — *UPQC: shunt active filter with voltage stabiliser*.
  Skeleton only (headings, no body).

## Hydraulics, heavy-duty vehicles and energy storage

- `hydraulic_fluid_power` — *Hydrostatic Fluid Power: modelization and control* (October 2022).
  Fluid modelling, hydrostatic components (pumps, motors, relief valve, pressure-controlled pump,
  spool valves), powertrain identification and validation against tests, servo actuator, solenoid
  valve, diesel engine model and control.
- `electrification_heavy_duty_vehicles` — *Electrification of Heavy Duty Vehicles* (October
  2022). Replacing the diesel and hydrostatic transmissions of a tracked bulldozer: hydrostatic
  powertrain model, PMSM design alternatives (16p/12s, 8p/12s) with thermal model and losses,
  and an ANSYS RMxprt/Maxwell tutorial chapter.
- `nonlinear_observers` — *Nonlinear Observers: application to hydrostatic transmission and to
  electrical machines* (October 2022). Load and driveline-pressure estimators for a hydrostatic
  transmission; appendices on DC motor, induction machine, nonlinear normal forms and
  local/EKF/global observers (Khalil, Isidori, Slotine).
- `lithium_ion_battery` — *Lithium Ion Battery: modelization and Simscape implementation*
  (October 2022). Thevenin equivalent circuit with SOC-dependent OCV, coulomb-counting SOC
  estimator, parameters and outputs of the Simscape block.
- `pemfuelcell` — *PEM Fuel Cell: model derivation and Simscape implementation* (October 2022).
  Nernst-based stack model, gas dynamics, material conservation, parameters and outputs.

## Notes

Several folders overlap: `igbt_selection_and_procedure` and the report in
`semiconductor_devices_igbt_1` share the same body; the three single-phase inverter folders share
the same body; `psm_mpc_ekf` reuses the title of `im_mpc_ekf`; `control-of-string-actuated`
reuses the abstract of `flexible_shaft`. The `ps_25V_12kA` title says "PS-12V-12kA".
