% Deterministic Gorodnichenko-Lee application fixtures (Section 5 and
% Appendix H2). Run from any directory with
%   octave --no-gui test/fixtures/goro_lee/drivers/app_deterministic.m
% Requires the gitignored reference tree docs/src/code/goro_lee_r2.
1;
addpath(fileparts(mfilename('fullpath')));
[root, ref, outdir] = app_setup();
[C, raw] = app_cases();
H = 20;
hcol = (0:H)';

% --- raw data and estimator inputs -------------------------------------
app_write_csv(fullfile(outdir, 'app_data_sec5.csv'), ...
    {'time','dtfp_util','rr_shock','dgdp','dcons','dinv','effr','infl'}, raw.sec5);
app_write_csv(fullfile(outdir, 'app_data_rz.csv'), ...
    {'time','news','dgdp','infl','unemp','tbill'}, raw.rz);
for i = 1:numel(C)
    c = C(i);
    app_write_csv(fullfile(outdir, ['app_' c.name '_inputs.csv']), ...
        [{'y'}, c.names], [c.y, c.X]);
end

% --- VAR(4) Cholesky point estimates (AppSimulation.m #3, var_chol) ------
hdr = {'h'};
M = hcol;
for shock = [1, 4]
    [psi, ~, ~, vd] = var_chol(raw.sec5_timeseries, 4, H, 2000, shock, 0, 2, 0, 0);
    for v = 1:5
        hdr{end+1} = sprintf('psi_s%d_v%d', shock, v); M = [M, psi(v,:)'];
    end
    for v = 1:5
        hdr{end+1} = sprintf('vd_s%d_v%d', shock, v);  M = [M, vd(v,:)'];
    end
end
app_write_csv(fullfile(outdir, 'app_sec5_var4_chol.csv'), hdr, M);
[psi, ~, ~, vd] = var_chol(raw.rz_timeseries, 4, H, 2000, 1, 0, 2, 0, 0);
hdr = {'h'}; M = hcol;
for v = 1:4, hdr{end+1} = sprintf('psi_s1_v%d', v); M = [M, psi(v,:)']; end
for v = 1:4, hdr{end+1} = sprintf('vd_s1_v%d', v);  M = [M, vd(v,:)'];  end
app_write_csv(fullfile(outdir, 'app_rz_var4_chol.csv'), hdr, M);

% --- per case: R2, LP-A, LP-B, IC lag and bootstrap-DGP FEVD ------------
icrows = [];
for i = 1:numel(C)
    c = C(i);
    % R2 (SubR2 with extraOutputIndicator = 1, as in BootstrapVARR2 / SimulationAVarR2)
    [vd, fe, sePW, seWoPW, idxt, theta, GPW, GWoPW, effT] = ...
        SubR2(c.y, c.X, c.lagy, c.lagX, H, c.ylevel, c.trend, 1);
    app_write_csv(fullfile(outdir, ['app_' c.name '_r2.csv']), ...
        {'h','t0','tend','Th','vd','se_pw','se_nopw'}, ...
        [hcol, idxt, effT, vd(:), sePW(:), seWoPW(:)]);
    if strcmp(c.name, 'tfp_y')
        app_write_csv(fullfile(outdir, 'app_tfp_y_r2_fe.csv'), ...
            arrayfun(@(h) sprintf('h%d', h), hcol', 'UniformOutput', false), fe);
        TH = []; GG = [];
        for h = 0:H
            th = theta{h+1};
            TH = [TH; repmat(h, numel(th), 1), (1:numel(th))', th];
            m = numel(th);
            [jj, ii] = meshgrid(1:m, 1:m);
            GG = [GG; repmat(h, m*m, 1), ii(:), jj(:), GPW{h+1}(:), GWoPW{h+1}(:)];
        end
        app_write_csv(fullfile(outdir, 'app_tfp_y_r2_theta.csv'), {'h','i','theta'}, TH);
        app_write_csv(fullfile(outdir, 'app_tfp_y_r2_gogt.csv'), {'h','i','j','gogt_pw','gogt_nopw'}, GG);
    end

    % LP-A / LP-B, VAR-bootstrap set (XLP = X; as in BootstrapVARLP*.m)
    [ir, vdA, ~, irse, meanx, varx, mse, ~, idxt, ~, ~, ~, ~, t0LP] = ...
        SubLPA(c.y, c.X, c.lagy, c.lagX, c.XtIndicator, H, c.ylevel, c.trend, c.dfIndicator, 1, 0);
    app_write_csv(fullfile(outdir, ['app_' c.name '_lpa_varset.csv']), ...
        {'h','t0','tend','t0lp','ir','irse','vd','mse','meanx','varx'}, ...
        [hcol, idxt, repmat(t0LP, H+1, 1), ir, irse, vdA, mse, repmat([meanx, varx], H+1, 1)]);
    [ir, vdB, ~, irse, meanx, varx, mse, ~, idxt, ~, ~, ~, ~, ~, s2v, t0LP] = ...
        SubLPB(c.y, c.X, c.lagy, c.lagX, c.XtIndicator, H, c.ylevel, c.trend, c.dfIndicator, 1, 0);
    app_write_csv(fullfile(outdir, ['app_' c.name '_lpb_varset.csv']), ...
        {'h','t0','tend','t0lp','ir','irse','vd','mse','sigma2v','meanx','varx'}, ...
        [hcol, idxt, repmat(t0LP, H+1, 1), ir, irse, vdB, mse, s2v, repmat([meanx, varx], H+1, 1)]);

    % LP-A / LP-B, separate parsimonious LP set (as in SimulationAVarLP*.m)
    [ir, vdA, ~, irse, meanx, varx, mse, ~, idxt, ~, ~, ~, ~, t0LP] = ...
        SubLPA(c.y, c.X, c.lagy, c.lagX, c.XtIndicator, H, c.ylevel, c.trend, c.dfIndicator, 1, 0, ...
               c.X, c.lagyLP, c.lagXLP, c.XtIndicator);
    app_write_csv(fullfile(outdir, ['app_' c.name '_lpa_lpset.csv']), ...
        {'h','t0','tend','t0lp','ir','irse','vd','mse','meanx','varx'}, ...
        [hcol, idxt, repmat(t0LP, H+1, 1), ir, irse, vdA, mse, repmat([meanx, varx], H+1, 1)]);
    [ir, vdB, ~, irse, meanx, varx, mse, ~, idxt, ~, ~, ~, ~, ~, s2v, t0LP] = ...
        SubLPB(c.y, c.X, c.lagy, c.lagX, c.XtIndicator, H, c.ylevel, c.trend, c.dfIndicator, 1, 0, ...
               c.X, c.lagyLP, c.lagXLP, c.XtIndicator);
    app_write_csv(fullfile(outdir, ['app_' c.name '_lpb_lpset.csv']), ...
        {'h','t0','tend','t0lp','ir','irse','vd','mse','sigma2v','meanx','varx'}, ...
        [hcol, idxt, repmat(t0LP, H+1, 1), ir, irse, vdB, mse, s2v, repmat([meanx, varx], H+1, 1)]);

    % VAR system, HQIC lag over 1:10 and bootstrap-DGP FEVD (pseudo-truth)
    o = app_varic(c.y, c.X, H, c.ylevel, c.trend, c.lagVARSet, c.XVAROrder, c.yPlace, c.ICIndicator);
    app_write_csv(fullfile(outdir, ['app_' c.name '_svar.csv']), ...
        {'h','lagvar','svar'}, [hcol, repmat(o.lagVAR, H+1, 1), o.vdVAR]);
    icrows = [icrows; repmat(i, numel(o.IC), 1), c.lagVARSet(:), ...
              repmat([size(o.XVAR, 1), o.T_IC, o.xLocation, o.yLocation], numel(o.IC), 1), ...
              o.IC, double(c.lagVARSet(:) == o.lagVAR)];
    printf('%-7s HQIC lag = %d (XVAR %d x %d, T_IC = %d)\n', c.name, o.lagVAR, size(o.XVAR), o.T_IC);
end
app_write_csv(fullfile(outdir, 'app_ic.csv'), ...
    {'case','p','xvar_rows','T_IC','x_location','y_location','hqic','selected'}, icrows);
printf('done: %s\n', outdir);
