% Stochastic Gorodnichenko-Lee application fixtures, TFP -> y (Section 5),
% paper settings, one fixed Octave seed per run. Usage:
%   octave --no-gui test/fixtures/goro_lee/drivers/app_stochastic.m SEED [NBOOT]
% Every procedure call is preceded by rand("state", SEED); randn("state", SEED)
% so each output file is reproducible on its own (Octave 11.3 Mersenne
% twister; MATLAB draws differ, so compare only up to Monte Carlo error).
1;
function reseed(s)
rand('state', s);
randn('state', s);
end
addpath(fileparts(mfilename('fullpath')));
[root, ref, outdir] = app_setup();
args = argv();
if numel(args) < 1
    error('usage: app_stochastic.m SEED [NBOOT]');
end
seed = str2double(args{1});
NB = 2000;
if numel(args) >= 2
    NB = str2double(args{2});
end
C = app_cases();
c = C(strcmp({C.name}, 'tfp_y'));
H = c.horizon;
hcol = (0:H)';
q = [0.05, 0.95];
TBurnIn = 100;
sizeBlock = 4;
seIndicator = 0; cbIndicator = 0; boundIndicator = 1; jointIndicator = 0;
o = app_varic(c.y, c.X, H, c.ylevel, c.trend, c.lagVARSet, c.XVAROrder, c.yPlace, c.ICIndicator);
svar = o.vdVAR;
tag = sprintf('tfp_y_seed%d', seed);
meta = {};

% 1. VAR residual bootstrap, R2
reseed(seed); tic;
[vd, vdBc, vdSe, cb, d] = BootstrapVARR2(c.y, c.X, H, c.ylevel, c.trend, c.lagy, c.lagX, ...
    c.lagVARSet, c.XVAROrder, c.yPlace, c.ICIndicator, NB, TBurnIn, q, seIndicator, cbIndicator);
t = toc; meta(end+1, :) = {'var_r2', t, d.lagVAR};
app_write_csv(fullfile(outdir, ['app_stoch_var_r2_' tag '.csv']), ...
    {'h','vd','vdbc','se_boot','cb05','cb95','mean_draws','svar','lagvar','se_avar_pw','se_avar_nopw'}, ...
    [hcol, vd(:), vdBc, vdSe, cb', vd(:) - vdBc + svar, svar, repmat(d.lagVAR, H+1, 1), d.vdSeAVarWPW(:), d.vdSeAVarWoPW(:)]);

% 2. VAR residual bootstrap, LP-B
reseed(seed); tic;
[vd, vdBc, vdSe, cb, d] = BootstrapVARLPB(c.y, c.X, H, c.ylevel, c.trend, c.lagy, c.lagX, c.XtIndicator, ...
    c.lagVARSet, c.XVAROrder, c.yPlace, c.ICIndicator, NB, TBurnIn, q, c.dfIndicator);
t = toc; meta(end+1, :) = {'var_lpb', t, d.lagVAR};
app_write_csv(fullfile(outdir, ['app_stoch_var_lpb_' tag '.csv']), ...
    {'h','vd','vdbc','se_boot','cb05','cb95','mean_draws','svar','lagvar', ...
     'ir','irbc','irse_boot','irse_avar','stdx','irstr','irstr_cb05','irstr_cb95'}, ...
    [hcol, vd(:), vdBc, vdSe, cb', vd(:) - vdBc + svar, svar, repmat(d.lagVAR, H+1, 1), ...
     d.ir, d.irBc, d.irSeBootstrap, d.irSeAVar, repmat(d.stdx, H+1, 1), d.irStr, d.irStrCb']);

% 3. Block bootstrap (Kilian-Kim), R2; defaults sizeBlock = 4, TBurnIn = 100
reseed(seed); tic;
[vd, vdBc, vdSe, cb, d] = BootstrapBlockR2(c.y, c.X, H, c.ylevel, c.trend, c.lagy, c.lagX, sizeBlock, ...
    NB, TBurnIn, q, seIndicator, cbIndicator);
t = toc; meta(end+1, :) = {'block_r2', t, NaN};
app_write_csv(fullfile(outdir, ['app_stoch_block_r2_' tag '.csv']), ...
    {'h','vd','vdbc','se_boot','cb05','cb95','mean_draws','se_avar_pw','se_avar_nopw'}, ...
    [hcol, vd(:), vdBc, vdSe, cb', 2 * vd(:) - vdBc, d.vdSeAVarWPW(:), d.vdSeAVarWoPW(:)]);

% 4. Asymptotic simulation, R2 (boundIndicator = 1)
reseed(seed); tic;
[vd, vdBc, vdSe, cb, d] = SimulationAVarR2(c.y, c.X, H, c.ylevel, c.trend, c.lagy, c.lagX, ...
    NB, q, seIndicator, cbIndicator, boundIndicator);
t = toc; meta(end+1, :) = {'sim_r2', t, NaN};
app_write_csv(fullfile(outdir, ['app_stoch_sim_r2_' tag '.csv']), ...
    {'h','vd','vdbc','se_boot','cb05','cb95','mean_draws','se_avar_pw_at_vdbc','se_avar_nopw_at_vdbc'}, ...
    [hcol, vd(:), vdBc, vdSe, cb', 2 * vd(:) - vdBc, d.vdSeAVarWPW(:), d.vdSeAVarWoPW(:)]);

% 5. Asymptotic simulation, LP-B, parsimonious LP set (lagyLP = 1, lagXLP = [1,0,0,0,0])
reseed(seed); tic;
[vd, vdBc, vdSe, cb, d] = SimulationAVarLPB(c.y, c.X, H, c.ylevel, c.trend, c.lagy, c.lagX, c.XtIndicator, ...
    NB, c.dfIndicator, q, seIndicator, cbIndicator, jointIndicator, c.X, c.lagyLP, c.lagXLP, c.XtIndicator);
t = toc; meta(end+1, :) = {'sim_lpb', t, NaN};
app_write_csv(fullfile(outdir, ['app_stoch_sim_lpb_' tag '.csv']), ...
    {'h','vd','vdbc','se_boot','cb05','cb95','mean_draws','se_avar_pw_at_vdbc','se_avar_nopw_at_vdbc', ...
     'ir','irse_avar','stdx'}, ...
    [hcol, vd(:), vdBc, vdSe, cb', 2 * vd(:) - vdBc, d.vdSeAVarWPW(:), d.vdSeAVarWoPW(:), ...
     d.ir, d.irSeAVar, repmat(d.stdx, H+1, 1)]);

for i = 1:rows(meta)
    printf('%-9s seed=%d nboot=%d seconds=%.1f lagVAR=%g\n', meta{i, 1}, seed, NB, meta{i, 2}, meta{i, 3});
end
