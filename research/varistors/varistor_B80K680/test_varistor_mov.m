% test_varistor_mov.m
% Static V-I sanity check of the varistor_mov parameterization against the
% two EPCOS B80K680 datasheet points. Pure MATLAB (no Simscape needed):
% reproduces the same equations used in varistor_mov.ssc at DC.

clear; clc;

% --- Parameters (defaults of varistor_mov.ssc = B80K680) ---
V1 = 1100;   I1 = 1e-3;   % varistor voltage @ 1 mA
V2 = 1815;   I2 = 800;    % clamping point @ 800 A (8/20 us)
Rp = 100e6;               % leakage resistance
Rb = 1e-3;                % series bulk resistance

alpha = log(I2/I1)/log(V2/V1);
fprintf('alpha = %.2f\n', alpha);

% --- Static characteristic (DC: C and Ls play no role) ---
vc  = logspace(log10(600), log10(2400), 500);
inl = I1*(vc/V1).^alpha;
ip  = vc/Rp;
ii  = inl + ip;
vv  = vc + Rb*ii;

% --- Check the two fit points ---
v_at_I1 = interp1(ii, vv, I1);
v_at_I2 = interp1(ii, vv, I2);
fprintf('v(%g A)  = %.1f V  (datasheet %g V)\n', I1, v_at_I1, V1);
fprintf('v(%g A)   = %.1f V  (datasheet %g V)\n', I2, v_at_I2, V2);

% --- Plot ---
figure;
loglog(ii, vv, 'b', 'LineWidth', 1.2); grid on; hold on;
loglog([I1 I2], [V1 V2], 'ro', 'MarkerFaceColor', 'r');
xlabel('i (A)'); ylabel('v (V)');
title(sprintf('varistor\\_mov static characteristic, \\alpha = %.1f', alpha));
legend('model', 'datasheet points', 'Location', 'northwest');
