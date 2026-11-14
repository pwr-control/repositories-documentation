%% planar_xfmr_design.m
% Dimensionamento trasformatore planare
% Specifiche: Vin=12V, Iin=0.5A, Vout=25V, Viso=12kV
% Topologia: Flyback / Forward, frequenza variabile
%
% Autore: Davide
% -------------------------------------------------------------------------

clear; close all; clc;

%% =========================================================================
%  SPECIFICHE
% =========================================================================
Vin      = 12;       % [V]  tensione ingresso
Iin_nom  = 0.5;      % [A]  corrente ingresso nominale
Vout     = 25;       % [V]  tensione uscita
Viso     = 12e3;     % [V]  isolamento richiesto
eta      = 0.88;     % [-]  rendimento stimato
D_max    = 0.45;     % [-]  duty-cycle massimo (flyback CCM)

Pin  = Vin * Iin_nom;            % [W]  potenza ingresso
Pout = Pin * eta;                % [W]  potenza uscita
n    = Vout / Vin;               % [-]  rapporto di trasformazione ideale
Iout = Pout / Vout;              % [A]  corrente uscita

fprintf('=== SPECIFICHE ===\n');
fprintf('  Pin  = %.2f W\n', Pin);
fprintf('  Pout = %.2f W\n', Pout);
fprintf('  n    = %.3f\n', n);
fprintf('  Iout = %.3f A\n', Iout);

%% =========================================================================
%  PARAMETRI CORE — ELP32/6/20 (Ferroxcube/TDK)
% =========================================================================
Ae   = 79e-6;    % [m^2] sezione effettiva (ELP32)
Aw   = 35e-6;    % [m^2] area finestra avvolgimento
Bmax = 0.15;     % [T]   induzione massima @ 100°C, N87/3F36
Ku   = 0.40;     % [-]   fattore utilizzo finestra

%% =========================================================================
%  SWEEP FREQUENZA
% =========================================================================
fsw_vec = (50:10:500) * 1e3;   % [Hz] da 50 kHz a 500 kHz

Np_vec      = zeros(size(fsw_vec));
Ns_vec      = zeros(size(fsw_vec));
Bpeak_vec   = zeros(size(fsw_vec));
Jrms_p_vec  = zeros(size(fsw_vec));
WaAe_vec    = zeros(size(fsw_vec));

for k = 1:length(fsw_vec)
    fsw = fsw_vec(k);

    % Spire primario — equazione di Faraday (flyback, onda quadra)
    %   Vp * D = Np * Ae * dB/dt  =>  Np = Vin*D / (4*fsw*Bmax*Ae)
    %   (fattore 4 per ciclo completo, forma d'onda simmetrica)
    Np = ceil( (Vin * D_max) / (4 * fsw * Bmax * Ae) );
    Ns = ceil( n * Np );

    % Verifica B_peak effettivo
    Bpeak = (Vin * D_max) / (4 * fsw * Np * Ae);

    % Correnti RMS stimate (DCM/CCM flyback approssimato)
    Ip_rms = Pin / (Vin * eta);
    Is_rms = Pout / Vout;

    % Area conduttore disponibile per spira (fattore Ku / spire totali)
    N_tot = Np + Ns;
    Aw_per_turn = Ku * Aw / N_tot;  % [m^2]

    % Densità di corrente primario [A/m^2]
    if Aw_per_turn > 0
        Jrms_p = Ip_rms / Aw_per_turn;
    else
        Jrms_p = NaN;
    end

    % Area prodotto Wa*Ae (core utilization)
    WaAe = Aw * Ae;

    Np_vec(k)     = Np;
    Ns_vec(k)     = Ns;
    Bpeak_vec(k)  = Bpeak;
    Jrms_p_vec(k) = Jrms_p * 1e-6;  % converti in A/mm^2
    WaAe_vec(k)   = WaAe * 1e8;     % converti in cm^4
end

%% =========================================================================
%  PUNTI OPERATIVI A 200 kHz
% =========================================================================
fsw_des = 200e3;
Np_des  = ceil( (Vin * D_max) / (4 * fsw_des * Bmax * Ae) );
Ns_des  = ceil( n * Np_des );
Bpk_des = (Vin * D_max) / (4 * fsw_des * Np_des * Ae);
Ip_rms  = Pin / (Vin * eta);
Is_rms  = Pout / Vout;

fprintf('\n=== PUNTO DI PROGETTO @ fsw = %.0f kHz ===\n', fsw_des/1e3);
fprintf('  Np     = %d spire\n', Np_des);
fprintf('  Ns     = %d spire\n', Ns_des);
fprintf('  n_eff  = %.3f\n', Ns_des/Np_des);
fprintf('  B_peak = %.3f T\n', Bpk_des);
fprintf('  Ip_rms = %.3f A\n', Ip_rms);
fprintf('  Is_rms = %.3f A\n', Is_rms);

%% =========================================================================
%  CALCOLO ISOLAMENTO — richiesta di traccia/layer
% =========================================================================
% Rigidita` dielettrica FR4 standard: ~20 kV/mm
% Kapton 25 um/layer: ~8 kV (soglia pratica 300 V/um)
k_FR4    = 20;      % [kV/mm]
k_kapton = 0.3;     % [kV/um] = 300 V/um

t_FR4_min    = Viso / (k_FR4 * 1e3);    % [mm]
t_kapton_min = Viso / (k_kapton * 1e6) * 1e6; % [um]

% Clearance IEC 62368 (OVC III, pollution degree 2)
clearance_mm = 14;    % [mm]
creepage_mm  = 28;    % [mm] CTI IIIa (FR4 ~ CTI 175)

fprintf('\n=== ISOLAMENTO 12 kV ===\n');
fprintf('  Spessore FR4 min (20 kV/mm):    %.2f mm\n', t_FR4_min);
fprintf('  Spessore Kapton min (300V/um):  %.0f um  (~%d strati da 25um)\n', ...
        t_kapton_min, ceil(t_kapton_min/25));
fprintf('  Clearance richiesta (IEC):      %.0f mm\n', clearance_mm);
fprintf('  Creepage richiesta (IEC/FR4):   %.0f mm\n', creepage_mm);

%% =========================================================================
%  LARGHEZZA TRACCIA PCB (IPC-2221)
% =========================================================================
% Formula semplificata: I = k * dT^0.44 * A^0.725
% Conduttore esterno: k=0.048 (A in mil^2, I in A)
% Conduttore interno: k=0.024
Cu_oz   = 2;           % [oz] spessore rame (2 oz = 70 um)
dT_max  = 10;          % [°C] riscaldamento massimo accettabile

k_ext = 0.048;
k_int = 0.024;

% Area in mil^2 per corrente data
A_p_ext = (Ip_rms / (k_ext * dT_max^0.44))^(1/0.725);  % [mil^2]
A_s_ext = (Is_rms / (k_ext * dT_max^0.44))^(1/0.725);

% Spessore rame in mil: 2oz = 2.8 mil = 71 um
t_cu_mil = 2.8 * Cu_oz;

% Larghezza
w_p_mm = (A_p_ext / t_cu_mil) * 0.0254;  % mil -> mm
w_s_mm = (A_s_ext / t_cu_mil) * 0.0254;

fprintf('\n=== TRACCE PCB (Cu 2 oz, dT=10°C) ===\n');
fprintf('  Larghezza primario (Ip=%.3fA):  %.3f mm\n', Ip_rms, w_p_mm);
fprintf('  Larghezza secondario (Is=%.3fA): %.3f mm\n', Is_rms, w_s_mm);

%% =========================================================================
%  FIGURE
% =========================================================================
fsw_kHz = fsw_vec / 1e3;

fig1 = figure('Name','Spire vs Frequenza','Position',[100 100 900 600]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile
plot(fsw_kHz, Np_vec, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 3); hold on
plot(fsw_kHz, Ns_vec, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 3);
xline(fsw_des/1e3, '--k', sprintf('f_{sw}=%d kHz', fsw_des/1e3), ...
      'LabelVerticalAlignment','bottom','LineWidth',1);
xlabel('$f_{sw}$ [kHz]', 'Interpreter', 'latex');
ylabel('Numero di spire', 'Interpreter', 'latex');
title('Spire primario e secondario', 'Interpreter', 'latex');
legend('$N_p$', '$N_s$', 'Interpreter', 'latex', 'Location', 'northeast');
grid on; box on;

nexttile
plot(fsw_kHz, Bpeak_vec * 1e3, 'Color', [0.2 0.6 0.2], 'LineWidth', 1.5); hold on
yline(Bmax * 1e3, '--r', '$B_{max}$', 'Interpreter', 'latex', ...
      'LabelVerticalAlignment','bottom', 'LineWidth', 1);
xline(fsw_des/1e3, '--k', '', 'LineWidth', 1);
xlabel('$f_{sw}$ [kHz]', 'Interpreter', 'latex');
ylabel('$\hat{B}$ [mT]', 'Interpreter', 'latex');
title('Induzione di picco', 'Interpreter', 'latex');
grid on; box on;

nexttile
plot(fsw_kHz, Jrms_p_vec, 'Color', [0.8 0.2 0.8], 'LineWidth', 1.5); hold on
yline(5, '--r', '$J_{max}=5\ \mathrm{A/mm^2}$', 'Interpreter', 'latex', ...
      'LabelVerticalAlignment','bottom', 'LineWidth', 1);
xline(fsw_des/1e3, '--k', '', 'LineWidth', 1);
xlabel('$f_{sw}$ [kHz]', 'Interpreter', 'latex');
ylabel('$J_{rms,p}$ [A/mm$^2$]', 'Interpreter', 'latex');
title('Densit\`{a} di corrente primario', 'Interpreter', 'latex');
grid on; box on;

nexttile
% Margine dielettrico: B_breakdown di FR4 con 2mm strato
t_iso_vec  = linspace(0.5, 4, 100);  % mm
Vbd_FR4    = t_iso_vec * k_FR4;      % kV
plot(t_iso_vec, Vbd_FR4, 'b-', 'LineWidth', 1.5); hold on
yline(Viso/1e3, '--r', sprintf('$V_{iso}=%.0f$ kV', Viso/1e3), ...
      'Interpreter', 'latex', 'LabelVerticalAlignment','bottom', 'LineWidth', 1);
xline(t_FR4_min, '--k', sprintf('$t_{min}=%.2f$ mm', t_FR4_min), ...
      'Interpreter', 'latex', 'LabelHorizontalAlignment','right', 'LineWidth', 1);
xlabel('Spessore FR4 [mm]', 'Interpreter', 'latex');
ylabel('$V_{breakdown}$ [kV]', 'Interpreter', 'latex');
title('Tensione di rottura FR4 (20 kV/mm)', 'Interpreter', 'latex');
grid on; box on;

% Export figura
set(fig1, 'PaperUnits', 'centimeters', 'PaperSize', [22 15], ...
    'PaperPosition', [0.5 0.5 21 14]);
print(fig1, '/home/claude/planar_xfmr/fig_sweep.pdf', '-dpdf', '-r300');
print(fig1, '/home/claude/planar_xfmr/fig_sweep.png', '-dpng', '-r200');
fprintf('\nFigure salvate in fig_sweep.pdf e fig_sweep.png\n');
