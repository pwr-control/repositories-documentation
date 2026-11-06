function upqc_summary(matfile, prefix, waveforms)
%UPQC_SUMMARY  Compact text summary of a simulation of ls_upqc.slx (figures and numbers of the design note).
%
%   upqc_summary(matfile, prefix, waveforms)
%     matfile    results: the logged *_sim arrays and param (save sim_results_2 after the simulation)
%     prefix     output files: <prefix>_env.csv (one row every 10 ms, quantities over the last grid period),
%                <prefix>_numbers.txt (averages over time windows, start-up figures)
%     waveforms  true: also <prefix>_wave.csv, waveforms from 0.72 s to 0.82 s every 0.2 ms
%
%   Needs iec_power_meter.m (folder ls_upqc) on the path.

    if nargin < 3
        waveforms = false;
    end
    S = load(matfile);
    p = S.param;
    t = arr(S.time_tc_sim);
    t = t(:);
    n = numel(t);
    dt = (t(end) - t(1))/(n - 1);
    f = p.grid.fgrid;
    np = round(1/(f*dt));                                   % samples in one grid period
    has = @(name) isfield(S, name);
    storage = has('voltage_battery_sim');

    ug = col3(S.ug_abc_sim, n);                             % [V]
    ig = col3(S.ig_abc_sim, n);                             % [A]
    ul = col3(S.uload_abc_pu_sim, n)*p.vs.hw.ubez;          % [V]
    il = col3(S.iload_abc_pu_sim, n)*p.vs.hw.ibez;          % [A]
    is_saf = col3(S.is_abc_pu_saf_sim, n)*p.saf.hw.ibez;    % [A]
    is_vs = col3(S.is_abc_pu_vs_sim, n)*p.vs.hw.ibez;       % [A]
    udc = col1(S.udc_saf_sim, n);                           % [V]
    pm_g = iec_power_meter(t, ug, ig, f);
    pm_l = iec_power_meter(t, ul, il, f);
    if storage
        ub = col1(S.voltage_battery_sim, n);
        ib = col1(S.current_battery_sim, n);
        udab = col1(S.dab_voltage_output_sim, n);
        idab = col1(S.dab_current_output_sim, n);
    end

    % ---- envelope: one row every 10 ms, quantities over the last grid period
    tk = (0.02:0.01:t(end)).';
    hdr = {'t', 'Ug1p', 'Ug1n', 'Ig1p', 'Ig1n', 'Pg', 'Qg', 'Pg1n', 'Qg1n', 'Ul1p', 'Ul1n', 'Pl', 'Ql', ...
        'THDug', 'THDul', 'Il_rms', 'Is_saf_rms', 'Is_vs_rms', 'Is_vs_rms_max', 'udc_mean', 'udc_min', 'udc_max'};
    if storage
        hdr = [hdr, {'ub_mean', 'ib_mean', 'pb_mean', 'pdab_out_mean'}];
    end
    E = nan(numel(tk), numel(hdr));
    for k = 1:numel(tk)
        i1 = min(n, round((tk(k) - t(1))/dt) + 1);
        i0 = i1 - np + 1;
        if i0 < 1
            continue
        end
        w = i0:i1;
        r = [tk(k), pm_g.U1p(i1), pm_g.U1n(i1), pm_g.I1p(i1), pm_g.I1n(i1), pm_g.P1p(i1), pm_g.Q1p(i1), ...
            pm_g.P1n(i1), pm_g.Q1n(i1), pm_l.U1p(i1), pm_l.U1n(i1), pm_l.P1p(i1), pm_l.Q1p(i1), ...
            thd(ug(w, 1)), thd(ul(w, 1)), mean(rms3(il(w, :))), mean(rms3(is_saf(w, :))), ...
            mean(rms3(is_vs(w, :))), max(rms3(is_vs(w, :))), mean(udc(w)), min(udc(w)), max(udc(w))];
        if storage
            r = [r, mean(ub(w)), mean(ib(w)), mean(ub(w).*ib(w)), mean(udab(w).*idab(w))]; %#ok<AGROW>
        end
        E(k, :) = r;
    end
    write_csv([prefix '_env.csv'], hdr, E);

    % ---- numbers
    fid = fopen([prefix '_numbers.txt'], 'w');
    [~, mname, mext] = fileparts(matfile);
    fprintf(fid, 'results %s | storage %d | logging step %.3g us | %d samples | grid period %d samples\n', ...
        [mname mext], storage, dt*1e6, n, np);
    fprintf(fid, 'fault %.3f s to %.3f s; reactive current step at %.3f s\n', p.frt.start_time_LVRT, ...
        p.frt.data.end_time_LVRT, p.saf.ref.time_i_react_pos_ref_1);
    win = {'pre-fault', 0.55, 0.75; 'fault', 1.20, 1.60; 'post-fault', 1.85, 1.99; 'after the Q step', 2.10, 2.25};
    for j = 1:size(win, 1)
        m = tk > win{j, 2} & tk <= win{j, 3};
        fprintf(fid, '\n[%s %.2f-%.2f s] mean (min, max) of the period values:\n', win{j, 1}, win{j, 2}, win{j, 3});
        for c = 2:numel(hdr)
            v = E(m, c);
            fprintf(fid, '  %-14s %12.5g  (%.5g, %.5g)\n', hdr{c}, mean(v), min(v), max(v));
        end
    end
    fprintf(fid, '\n[start-up]\n');
    fprintf(fid, '  max |ig| 0-0.1 s: %.4g A; max |ig| 0.1-0.3 s: %.4g A\n', max(max(abs(ig(t <= 0.1, :)))), ...
        max(max(abs(ig(t > 0.1 & t <= 0.3, :)))));
    for tt = [0.005 0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5]
        i = min(n, round((tt - t(1))/dt) + 1);
        fprintf(fid, '  t = %.3f s: udc = %.5g V\n', tt, udc(i));
    end
    i = find(udc > 0.99*p.dclink.udc_ref, 1);
    if ~isempty(i)
        fprintf(fid, '  udc above 0.99 udc_ref (%.0f V) from %.4f s\n', 0.99*p.dclink.udc_ref, t(i));
    end
    fprintf(fid, '  udc max overall %.5g V at %.4f s; min after 0.3 s %.5g V\n', max(udc), t(find(udc == max(udc), 1)), ...
        min(udc(t > 0.3)));
    fprintf(fid, '  max |is_saf| %.4g A, max |is_vs| %.4g A (after 0.3 s: %.4g A, %.4g A)\n', max(abs(is_saf(:))), ...
        max(abs(is_vs(:))), max(max(abs(is_saf(t > 0.3, :)))), max(max(abs(is_vs(t > 0.3, :)))));
    fprintf(fid, '  bases: SAF ubez %.4g V ibez %.4g A, VS ubez %.4g V ibez %.4g A\n', p.saf.hw.ubez, p.saf.hw.ibez, ...
        p.vs.hw.ubez, p.vs.hw.ibez);
    fclose(fid);

    % ---- waveforms around the start of the fault
    if waveforms
        m = find(t >= 0.72 & t <= 0.82);
        m = m(1:max(1, round(0.2e-3/dt)):end);
        W = [t(m), ug(m, :), ul(m, :), ig(m, :), udc(m)];
        write_csv([prefix '_wave.csv'], {'t', 'uga', 'ugb', 'ugc', 'ula', 'ulb', 'ulc', 'iga', 'igb', 'igc', 'udc'}, W);
    end
    fprintf('upqc_summary: %s -> %s_*\n', matfile, prefix);
end

function x = arr(x)
    if isa(x, 'timeseries')
        x = squeeze(x.Data);
    end
    x = double(x);
end

function x = col3(x, n)
    x = arr(x);
    if size(x, 1) ~= n
        x = x.';
    end
    assert(size(x, 1) == n, 'samples: %d, time: %d', size(x, 1), n);
end

function x = col1(x, n)
    x = arr(x);
    x = x(:);
    assert(numel(x) == n, 'samples: %d, time: %d', numel(x), n);
end

function r = rms3(x)
    r = sqrt(mean(x.^2, 1));
end

function y = thd(x)
% THD [%] of one grid period of samples, harmonics 2 to 50
    X = fft(x(:) - mean(x));
    h = min(50, floor(numel(x)/2) - 1);
    y = 100*sqrt(sum(abs(X(3:h + 1)).^2))/abs(X(2));
end

function write_csv(file, hdr, M)
    fid = fopen(file, 'w');
    fprintf(fid, '%s\n', strjoin(hdr, ','));
    fmt = [strjoin(repmat({'%.6g'}, 1, size(M, 2)), ','), '\n'];
    fprintf(fid, fmt, M.');
    fclose(fid);
end
