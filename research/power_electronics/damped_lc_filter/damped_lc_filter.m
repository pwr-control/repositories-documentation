%% damped_lc_filter.m
% L - C1 - (C2 + R2) output filter of an IGBT full bridge (unipolar PWM).
% Companion of "Passively Damped LC Output Filter with R-C Damping Branch",
% Rev. 00, 2026-09-21.  Requires the Control System Toolbox.
clear; clc; close all

%% ---- Input data (edit) -------------------------------------------------
Vdc   = 800;          % DC-link voltage [V]
In    = 500;          % rated output current [A]
fsw   = [1e3 10e3];   % switching frequency range per leg [Hz]
dVpp  = 0.01*Vdc;     % output voltage ripple spec, peak-to-peak [V]
delta = 0.20;         % max inductor current ripple, fraction of In
fc    = 30;           % current-loop crossover [Hz]
fac   = 5;            % max output fundamental [Hz]
Vac   = 400;          % max slow-AC output amplitude [V peak]
dvdt  = 2*pi*fac*600; % max output voltage slew rate [V/s]
Lload = 10e-3;  Rload = 0.1;      % load model
n     = 4;            % C2/C1
crit  = 'P';          % 'Z' Erickson (min peak Zo), 'H' (min peak H), 'P' (pole at -w0)
L     = 1.2e-3;       % chosen inductor [H]   (see step 3)
C1    = 680e-6;       % chosen capacitor [F]  (see step 4)

%% ---- Step 2: resonance frequency window -------------------------------
fr    = 2*fsw;                      % ripple frequency (unipolar PWM)
V1    = 2*Vdc/pi;                   % worst-case first harmonic (m = 0.5)
f0max = min(fr)*sqrt(dVpp/(2*V1));  % from the -40 dB/dec asymptote
f0min = 5*fc;                       % from the control bandwidth
fprintf('f0 window: %.0f ... %.0f Hz\n', f0min, f0max);

%% ---- Step 3: inductor ---------------------------------------------------
Lmin  = Vdc/(8*min(fsw)*delta*In);
dIpp  = Vdc./(8*fsw*L);             % worst-case ripple (m = 0.5)
fprintf('L >= %.2f mH, chosen %.2f mH -> dI_pp = %.1f / %.1f A\n', ...
        Lmin*1e3, L*1e3, dIpp);

%% ---- Step 4: capacitor and base quantities ------------------------------
w0 = 1/sqrt(L*C1);  f0 = w0/(2*pi);  R0 = sqrt(L/C1);
fprintf('f0 = %.1f Hz, R0 = %.3f ohm\n', f0, R0);

%% ---- Steps 5-6: damping branch -----------------------------------------
rZ = sqrt((2+n)*(4+3*n)/(2*n^2*(4+n)));   % Erickson, min peak |Zo|
rH = sqrt((1+n)*(2+n)/2)/n;               % min peak |H|
rP = (2+n)/(2*n);                         % real pole at -w0
switch crit
    case 'Z', r = rZ;
    case 'H', r = rH;
    otherwise, r = rP;
end
C2 = n*C1;  R2 = r*R0;
fprintf('n = %g: rZ = %.3f, rH = %.3f, rP = %.3f -> R2 = %.3f ohm, C2 = %.2f mF\n', ...
        n, rZ, rH, rP, R2, C2*1e3);
fprintf('closed form: peak|Zo| = %.3f ohm (rZ), peak|H| = %.3f (rH), fm = %.1f Hz\n', ...
        R0*sqrt(2*(2+n))/n, (2+n)/n, f0*sqrt(2/(2+n)));

%% ---- Transfer functions -------------------------------------------------
s   = tf('s');
Zc  = (1 + s*R2*C2)/(s*((C1+C2) + s*R2*C1*C2));   % shunt network
H   = minreal(Zc/(s*L + Zc));                      % vo/vb, unloaded
Zo  = minreal(s*L*H);                              % output impedance
Zin = minreal(s*L + Zc);                           % seen by the bridge
Zl  = Rload + s*Lload;
Zcl = minreal(Zc*Zl/(Zc + Zl));
Hl  = minreal(Zcl/(s*L + Zcl));                    % vo/vb, loaded
Hund = 1/(1 + s^2*L*C1);                           % undamped reference

[gH, wH] = getPeakGain(H);   [gZ, wZ] = getPeakGain(Zo);
fprintf('peak|H| = %.3f (%.1f dB) at %.0f Hz; peak|Zo| = %.3f ohm at %.0f Hz\n', ...
        gH, 20*log10(gH), wH/(2*pi), gZ, wZ/(2*pi));
damp(H)                                            % poles, zeta, frequencies

%% ---- Ripple check -------------------------------------------------------
for k = 1:numel(fr)
    Hk  = abs(freqresp(H, 2*pi*fr(k)));
    Vo1 = V1*Hk;                                  % first harmonic at output
    w   = 2*pi*fr(k);
    I2  = Vo1/sqrt(2)/abs(R2 + 1/(1j*w*C2));      % branch current, rms
    fprintf('fr = %5.0f Hz: |H| = %.1f dB, dVo_pp = %.2f V (%.2f %%), I2 = %.2f A rms, P_R2 = %.1f W\n', ...
            fr(k), 20*log10(Hk), 2*Vo1, 200*Vo1/Vdc, I2, I2^2*R2);
end
fprintf('triangular estimate: dVo_pp = %.2f V at fr,min\n', dIpp(1)/(8*fr(1)*C1));

%% ---- Slow-AC and slew losses in R2 -------------------------------------
wac  = 2*pi*fac;
I2ac = Vac/sqrt(2)*wac*C2/sqrt(1 + (wac*R2*C2)^2);
I1ac = Vac/sqrt(2)*wac*C1;
fprintf('%g Hz, %g V pk: I2 = %.1f A rms -> P_R2 = %.0f W; I_C1 = %.1f A rms\n', ...
        fac, Vac, I2ac, I2ac^2*R2, I1ac);
fprintf('slew %.1f kV/s: i2 = %.0f A pk, %.1f kW in R2 during the slew\n', ...
        dvdt/1e3, C2*dvdt, (C2*dvdt)^2*R2/1e3);

%% ---- Low-frequency behaviour (control) ---------------------------------
for f = [fac fc 2*fc]
    h = freqresp(H, 2*pi*f);
    fprintf('f = %4.0f Hz: |H| = %.3f, phase = %.1f deg\n', f, abs(h), angle(h)*180/pi);
end

%% ---- Sensitivity of the peaks to R2 -------------------------------------
kk = logspace(-1, 1, 41);  pkZ = zeros(size(kk));  pkH = pkZ;
for i = 1:numel(kk)
    R2i = kk(i)*rZ*R0;
    Zci = (1 + s*R2i*C2)/(s*((C1+C2) + s*R2i*C1*C2));
    Hi  = minreal(Zci/(s*L + Zci));
    pkH(i) = getPeakGain(Hi);  pkZ(i) = getPeakGain(minreal(s*L*Hi));
end

%% ---- Plots (grey scale) -------------------------------------------------
fv = logspace(0, 5, 2000);  w = 2*pi*fv;
figure('Name', 'Voltage transfer function');
semilogx(fv, 20*log10(squeeze(abs(freqresp(Hund, w)))), '-.', 'Color', [.55 .55 .55], 'LineWidth', 1.2); hold on
semilogx(fv, 20*log10(squeeze(abs(freqresp(H, w)))),    'k-',  'LineWidth', 1.6);
semilogx(fv, 20*log10(squeeze(abs(freqresp(Hl, w)))),   'k--', 'LineWidth', 1.2);
xline(fr(1), ':', 'Color', [.4 .4 .4]); xline(fr(end), ':', 'Color', [.4 .4 .4]);
grid on; ylim([-100 50]); xlabel('f (Hz)'); ylabel('|H| (dB)');
legend('undamped', 'damped, unloaded', 'damped, loaded', 'Location', 'southwest');

figure('Name', 'Output impedance');
semilogx(fv, 20*log10(squeeze(abs(freqresp(Zo, w)))), 'k-', 'LineWidth', 1.6); hold on
semilogx(fv, 20*log10(abs(1j*w*L)), ':', 'Color', [.5 .5 .5]);
semilogx(fv, 20*log10(1./(w*C1)), ':', 'Color', [.5 .5 .5]);
grid on; xlabel('f (Hz)'); ylabel('|Z_o| (dB ohm)'); ylim([-40 20]);
legend('|Z_o|', '\omega L', '1/(\omega C_1)', 'Location', 'south');

figure('Name', 'Peak sensitivity');
semilogx(kk, pkZ/min(pkZ), 'k-', kk, pkH/min(pkH), 'k--', 'LineWidth', 1.4);
grid on; xlabel('R_2 / R_{2,Z}'); ylabel('peak / minimum peak');
legend('|Z_o|', '|H|', 'Location', 'north');

figure('Name', 'Step response');
[y, t] = step(H, 0.06);  [yl, tl] = step(Hl, 0.06);
plot(t*1e3, y, 'k-', tl*1e3, yl, 'k--', 'LineWidth', 1.4); grid on
xlabel('t (ms)'); ylabel('v_o / \Delta v_b'); legend('unloaded', 'loaded');
S = stepinfo(H);
fprintf('step: overshoot %.1f %%, settling (2 %%) %.1f ms\n', S.Overshoot, S.SettlingTime*1e3);
