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

- `single_phase_grid_tied_inverter_analysis` — *Single Phase Inverter: model derivation and
  control architecture* (June 2020). Resonant PI, SOGI virtual-dq and harmonic-observer
  virtual-dq control compared on a unified state-space model with LC/LCL filter; results at 50,
  80 and 400 Hz.
- `single_phase_inverter` — *Single Phase Inverter: analysis of architecture and control
  strategies* (June 2026), same body updated.
- `three_phase_grid_tied_inverter_analysis` — despite the folder name, another copy of the
  single-phase inverter study.
- `single_phase_lock_loop` — *Single Phase Lock Loop*. SOGI-QSG based PLL and moving-average
  filter implemented on dSPACE SCALEXIO with C-Caller and DFT.
- `three_phase_inductors` — *Three Phase Inductors: mathematical model and construction*. Four-
  versus three-limb reactors (differential and common mode), saturation effects.
- `isop_architecture` — *Four-Winding Transformer: circuit model and governing equations*.
  Flux-linkage matrix, T-equivalent, MMF balance and port relations of the transformer used in
  the input-series output-parallel rail supply.
- `solid_state_transformers` — *Solid State Transformers: architecture, modelization and
  control* (May 2021). DAB, three-phase DAB and CLLC stages with NPC and T-type inverters,
  control loops, HF transformer design, simulation comparison.
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
