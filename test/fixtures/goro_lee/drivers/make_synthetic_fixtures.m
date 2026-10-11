% make_synthetic_fixtures.m -- deterministic synthetic fixtures for the
% Gorodnichenko & Lee (2020) FEVD port in LocalProjections.jl.
%
% Runs the read-only MATLAB reference code (docs/src/code/goro_lee_r2/lib,
% gitignored, never modified; added to the path with addpath) under GNU Octave
% 11.3 on synthetic data and writes every input and output as headerless CSV
% ('%.17g') under test/fixtures/goro_lee/synthetic/. See ../README.md.
%
% Usage (any cwd):
%   octave --no-gui test/fixtures/goro_lee/drivers/make_synthetic_fixtures.m
%
% Generate the verbatim-copy drivers first (they are not committed):
%   sh test/fixtures/goro_lee/drivers/gen_verbatim_drivers.sh

1; % script file marker (local functions below)

function wcell(dirn, stem, C)
% one CSV per horizon: <stem>_h<h>.csv for C{h+1}
for i = 1:numel(C)
    syn_writecsv(fullfile(dirn, sprintf('%s_h%d.csv', stem, i - 1)), C{i});
end
end

function d = mkd(varargin)
d = fullfile(varargin{:});
if ~exist(d, 'dir')
    mkdir(d);
end
end

function chk(name, v)
% finiteness check: abort if any NaN/Inf where finite values are expected
if iscell(v)
    for i = 1:numel(v)
        chk(sprintf('%s{%d}', name, i), v{i});
    end
    return
end
if ~all(isfinite(v(:)))
    error('non-finite values in %s', name);
end
if ~isreal(v)
    error('complex values in %s', name);
end
end

function chkwin(name, M, idxt)
% forecast-error style T x (H+1) matrix: finite inside rows idxt(h,1):idxt(h,2),
% NaN outside
for h = 1:size(M, 2)
    inside = false(size(M, 1), 1);
    inside(idxt(h, 1):idxt(h, 2)) = true;
    if ~all(isfinite(M(inside, h))) || ~all(isnan(M(~inside, h)))
        error('window pattern violated in %s, column %d', name, h);
    end
end
end

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', '..');
lib = fullfile(root, 'docs', 'src', 'code', 'goro_lee_r2', 'lib');
if ~exist(fullfile(lib, 'SubR2.m'), 'file')
    error('reference tree not found at %s', lib);
end
addpath(lib);
addpath(fullfile(here, '..', 'octave_shims'));
addpath(here);
out = mkd(here, '..', 'synthetic');
printf('Octave %s; writing to %s\n', version, out);

% ------------------------------------------------------------------ #0 data
% T = 200. x iid N(0,1) shock; z2, z3 stationary controls; dy stationary and
% driven by x; y = 10 + cumsum(dy) (levels). X = [x, z2, z3].
T = 200;
randn('state', 20200101);
E = randn(T, 4);
x = E(:, 1);
z2 = zeros(T, 1); z3 = zeros(T, 1); dy = zeros(T, 1);
for t = 1:T
    if t == 1
        dyl = 0; z2l = 0; z3l = 0; xl = 0;
    else
        dyl = dy(t-1); z2l = z2(t-1); z3l = z3(t-1); xl = x(t-1);
    end
    z2(t) = 0.5 * z2l + 0.3 * xl + E(t, 2);
    z3(t) = 0.4 * z3l + 0.2 * dyl + 0.5 * E(t, 3);
    dy(t) = 0.1 + 0.4 * dyl + 0.6 * x(t) + 0.3 * xl + 0.2 * z2l + 0.8 * E(t, 4);
end
y = 10 + cumsum(dy);
X = [x, z2, z3];
syn_writecsv(fullfile(out, 'data_y.csv'), y);
syn_writecsv(fullfile(out, 'data_X.csv'), X);

H = 8; lagy = 2; lagX = [4 2 3];

% ------------------------------------------------------------------ #1 SubR2
r2cases = { 'r2_yl0_tr0', 0, 0; 'r2_yl1_tr0', 1, 0; 'r2_yl0_tr1', 0, 1 };
for i = 1:size(r2cases, 1)
    [nm, ylevel, trend] = r2cases{i, :};
    d = mkd(out, nm);
    [vd, fe, seWPW, seWoPW, idxt, theta, GWPW, GWoPW, effT] = ...
        SubR2(y, X, lagy, lagX, H, ylevel, trend, 1);
    chk([nm ' vd'], vd); chk([nm ' se'], [seWPW, seWoPW]); chk([nm ' theta'], theta);
    chk([nm ' GOGT'], GWPW); chk([nm ' GOGT'], GWoPW); chkwin([nm ' fe'], fe, idxt);
    syn_writecsv(fullfile(d, 'vd.csv'), vd);
    syn_writecsv(fullfile(d, 'forecastError.csv'), fe);
    syn_writecsv(fullfile(d, 'seAVarWPW.csv'), seWPW);
    syn_writecsv(fullfile(d, 'seAVarWoPW.csv'), seWoPW);
    syn_writecsv(fullfile(d, 'idxt.csv'), idxt);
    syn_writecsv(fullfile(d, 'effectiveT.csv'), effT);
    wcell(d, 'theta', theta); wcell(d, 'GOGTWPW', GWPW); wcell(d, 'GOGTWoPW', GWoPW);
    printf('%s: vd = %s\n', nm, mat2str(vd', 4));
end

% ------------------------------------------------------------------ #2 SubLPA / SubLPB
XtInd = [1 1 0];
XLP = X; lagyLP = 1; lagXLP = [1 0 0]; XtIndLP = [1 0 0]; % separate (parsimonious) LP set
for ylevel = 0:1
    for df = 0:1
        for sep = 0:1
            setname = {'shared', 'sep'}{sep + 1};
            % LP-A
            nm = sprintf('lpa_yl%d_df%d_%s', ylevel, df, setname);
            d = mkd(out, nm);
            if sep
                [ir, vd, fe, irSe, meanx, varx, mse, ~, idxt, dhy, rhsLP, resLP, Beta, t0LP] = ...
                    SubLPA(y, X, lagy, lagX, XtInd, H, ylevel, 0, df, 1, 0, XLP, lagyLP, lagXLP, XtIndLP);
            else
                [ir, vd, fe, irSe, meanx, varx, mse, ~, idxt, dhy, rhsLP, resLP, Beta, t0LP] = ...
                    SubLPA(y, X, lagy, lagX, XtInd, H, ylevel, 0, df, 1, 0);
            end
            chk([nm ' scalars'], [ir; vd; irSe; meanx; varx; mse; Beta(:)]); chkwin([nm ' fe'], fe, idxt);
            names = {'ir', 'vd', 'forecastError', 'irSeAVar', 'meanx', 'varx', 'mse', 'idxt', 'dhy', 'rhsAllLP', 'residualLP', 'BetaHat', 't0LP'};
            vals = {ir, vd, fe, irSe, meanx, varx, mse, idxt, dhy, rhsLP, resLP, Beta, t0LP};
            for k = 1:numel(names)
                syn_writecsv(fullfile(d, [names{k} '.csv']), vals{k});
            end
            printf('%s: vd = %s\n', nm, mat2str(vd', 4));
            % LP-B
            nm = sprintf('lpb_yl%d_df%d_%s', ylevel, df, setname);
            d = mkd(out, nm);
            if sep
                [ir, vd, fe, irSe, meanx, varx, mse, ~, idxt, dhy, rhsLP, resLP, vFE, Beta, s2v, t0LP] = ...
                    SubLPB(y, X, lagy, lagX, XtInd, H, ylevel, 0, df, 1, 0, XLP, lagyLP, lagXLP, XtIndLP);
            else
                [ir, vd, fe, irSe, meanx, varx, mse, ~, idxt, dhy, rhsLP, resLP, vFE, Beta, s2v, t0LP] = ...
                    SubLPB(y, X, lagy, lagX, XtInd, H, ylevel, 0, df, 1, 0);
            end
            chk([nm ' scalars'], [ir; vd; irSe; meanx; varx; mse; s2v; Beta(:)]);
            chkwin([nm ' fe'], fe, idxt); chkwin([nm ' vFE'], vFE, idxt);
            names = {'ir', 'vd', 'forecastError', 'irSeAVar', 'meanx', 'varx', 'mse', 'idxt', 'dhy', 'rhsAllLP', 'residualLP', 'vFE', 'BetaHat', 'sigma2vHat', 't0LP'};
            vals = {ir, vd, fe, irSe, meanx, varx, mse, idxt, dhy, rhsLP, resLP, vFE, Beta, s2v, t0LP};
            for k = 1:numel(names)
                syn_writecsv(fullfile(d, [names{k} '.csv']), vals{k});
            end
            printf('%s: vd = %s\n', nm, mat2str(vd', 4));
        end
    end
end

% ------------------------------------------------------------------ #3 SimulationAVar* deterministic prefix
% R2 (paper settings, ylevel = 0, trend = 0)
d = mkd(out, 'simavar_r2_yl0');
o = sim_avar_r2_det(y, X, H, 0, 0, lagy, lagX);
chk('simavar_r2', [o.vd; o.seWPW; o.seWoPW]);
syn_writecsv(fullfile(d, 'vd.csv'), o.vd);
syn_writecsv(fullfile(d, 'seAVarWPW.csv'), o.seWPW);
syn_writecsv(fullfile(d, 'seAVarWoPW.csv'), o.seWoPW);
wcell(d, 'theta', o.theta); wcell(d, 'GOGTWPW', o.GOGTWPW); wcell(d, 'GOGTWoPW', o.GOGTWoPW);
% LP-A / LP-B, parsimonious LP set, dfIndicator = 0, jointIndicator = 0
for M = 'AB'
    for ylevel = 0:1
        nm = sprintf('simavar_lp%s_yl%d', lower(M), ylevel);
        d = mkd(out, nm);
        f = str2func(['sim_avar_lp' M '_det']);
        o = f(y, X, H, ylevel, 0, lagy, lagX, XtInd, 0, XLP, lagyLP, lagXLP, XtIndLP);
        chk(nm, [o.vd; o.ir; o.irSeAVar; o.seWPW; o.seWoPW; o.forecastErrorVar]);
        chk([nm ' GOGT'], o.GOGTWPW); chk([nm ' GOGT'], o.GOGTWoPW); chk([nm ' Deltah'], o.Deltah);
        syn_writecsv(fullfile(d, 'vd.csv'), o.vd);
        syn_writecsv(fullfile(d, 'ir.csv'), o.ir);
        syn_writecsv(fullfile(d, 'irSeAVar.csv'), o.irSeAVar);
        syn_writecsv(fullfile(d, 'varx.csv'), o.varx);
        syn_writecsv(fullfile(d, 'forecastErrorVar.csv'), o.forecastErrorVar);
        syn_writecsv(fullfile(d, 'Sigmaq.csv'), o.Sigmaq);
        syn_writecsv(fullfile(d, 'Nq.csv'), o.Nq);
        syn_writecsv(fullfile(d, 't0LP.csv'), o.t0LP);
        syn_writecsv(fullfile(d, 'window.csv'), o.window);
        syn_writecsv(fullfile(d, 'seAVarWPW.csv'), o.seWPW);
        syn_writecsv(fullfile(d, 'seAVarWoPW.csv'), o.seWoPW);
        wcell(d, 'theta', o.theta); wcell(d, 'Deltah', o.Deltah); % (Z_h not written: ~6 MB, rebuildable)
        wcell(d, 'GOGTWPW', o.GOGTWPW); wcell(d, 'GOGTWoPW', o.GOGTWoPW);
        printf('%s: seWPW = %s\n', nm, mat2str(o.seWPW', 4));
    end
end

% ------------------------------------------------------------------ #4 lrv_nw and prewhitened lrv
d = mkd(out, 'lrv');
n = 60;
randn('state', 7);
e = randn(n, 3);
Zm = zeros(n, 3);
for t = 1:n
    if t == 1
        Zm(t, :) = e(t, :);
    else
        Zm(t, :) = Zm(t-1, :) * [0.5 0.1 0; 0 0.3 0.2; 0.1 0 -0.2] + e(t, :);
    end
end
syn_writecsv(fullfile(d, 'Z.csv'), Zm);
for L = [0 1 3]
    V = lrv_nw(Zm, L);
    chk('lrv', V);
    syn_writecsv(fullfile(d, sprintf('lrv_L%d.csv', L)), V);
end
% prewhitened lrv: the operations of SubR2.m lines 214-220 applied to Zm
% (Zm is used as given, i.e. as the already-demeaned ZFull)
ZFull = Zm;
Lpw = round(0.75 * n^(1/3) - 1);
Z_1 = ZFull(2:end,:);
Z_0 = ZFull(1:end-1,:);
A = (Z_0' * Z_0) \ (Z_0' * Z_1);
A = A';
U = Z_1 - Z_0 * A';
OmegaWPW = (eye(3) - A) \ lrv_nw(U, Lpw) / (eye(3) - A');
chk('lrv pw', OmegaWPW);
syn_writecsv(fullfile(d, 'pw_A.csv'), A);
syn_writecsv(fullfile(d, 'pw_U.csv'), U);
syn_writecsv(fullfile(d, 'pw_L.csv'), Lpw);
syn_writecsv(fullfile(d, 'pw_Omega.csv'), OmegaWPW);
% bandwidth table L(T) = round(0.75*T^(1/3) - 1), T = 1..2000 (MATLAB round:
% half away from zero; '^' not nthroot)
Tg = (1:2000)';
syn_writecsv(fullfile(d, 'bandwidth.csv'), [Tg, round(0.75 * Tg.^(1/3) - 1)]);

% ------------------------------------------------------------------ #5 var_estimation / var_chol
d = mkd(out, 'var');
Yv = [x(2:end), diff(y), z2(2:end)]; % 3 variables, G-L convention rows 2:T
syn_writecsv(fullfile(d, 'input.csv'), Yv);
for di = 0:1
    if di
        dIdx = 2; tag = 'diff';
    else
        dIdx = []; tag = 'nodiff';
    end
    [psi, ~, ~, vdv, ~, ~, ~, c, phi, omega, psiRed, k, Teff, res] = ...
        var_chol(Yv, 2, H, 0, 1, 0, dIdx, 0, 0);
    chk('var_chol', [vdv(:); psi(:); c(:); phi(:); omega(:); res(:)]);
    syn_writecsv(fullfile(d, ['vd_' tag '.csv']), vdv);
    syn_writecsv(fullfile(d, ['psi_' tag '.csv']), psi);
    printf('var_chol %s: vd(2,:) = %s\n', tag, mat2str(vdv(2, :), 4));
end
syn_writecsv(fullfile(d, 'c.csv'), c);
syn_writecsv(fullfile(d, 'phi.csv'), reshape(phi, 3, 3 * 2));
syn_writecsv(fullfile(d, 'omega.csv'), omega);
syn_writecsv(fullfile(d, 'residual.csv'), res);
syn_writecsv(fullfile(d, 'k_T.csv'), [k, Teff]);
syn_writecsv(fullfile(d, 'psiReduced.csv'), reshape(psiRed, 3, 3 * (H + 1)));

% ------------------------------------------------------------------ #6 IC block of BootstrapVARR2.m
d = mkd(out, 'ic');
icnames = {'bic', 'hqic', 'aic'};
for ylevel = 0:1
    for ICI = 0:2
        [IC, lagVAR, XVAR, Tic] = ic_block(y, X, ylevel, 0, 1:6, [1 2 3], 2, ICI);
        chk('IC', IC);
        syn_writecsv(fullfile(d, sprintf('ic_yl%d_%s.csv', ylevel, icnames{ICI + 1})), IC(:));
        syn_writecsv(fullfile(d, sprintf('lag_yl%d_%s.csv', ylevel, icnames{ICI + 1})), [lagVAR, Tic]);
        printf('IC yl%d %s: lag %d, IC = %s\n', ylevel, icnames{ICI + 1}, lagVAR, mat2str(IC, 6));
    end
    syn_writecsv(fullfile(d, sprintf('XVAR_yl%d.csv', ylevel)), XVAR);
end

% ------------------------------------------------------------------ #7 var_bootstrap with recorded draws
d = mkd(out, 'varboot');
TBurnIn = 100;
% case 1: the 3-variable VAR(2) of #5 (Ty - 1 = 199 rows, Tout = Ty = 200)
% case 2: the 4-variable levels XVAR of #6 (ylevel = 1, 200 rows), VAR(1)
[~, ~, XVAR1] = ic_block(y, X, 1, 0, 1, [1 2 3], 2, 1);
bcases = {Yv, 2, 'case1'; XVAR1, 1, 'case2'};
for i = 1:2
    [Yb, p, tag] = bcases{i, :};
    [c, phi, omega, ~, ~, Teff, res] = var_estimation(Yb, p, 0, 0);
    Tend = TBurnIn + T;
    seed = 4242 + i;
    rand('twister', seed);
    idx = ceil(size(res, 2) * rand(Tend, 1));
    init = ceil((size(Yb, 1) - p + 1) * rand(1));
    rand('twister', seed);
    Ysim = var_bootstrap(c, phi, Yb, res, T, TBurnIn);
    % replay with the recorded draws (MATLAB-order loop of var_bootstrap.m)
    n = size(Yb, 2);
    F = [reshape(phi, n, n * p); eye(n * (p - 1)), zeros(n * (p - 1), n)];
    z = Yb(init + p - 1:-1:init, :)'; z = z(:);
    zb = zeros(Tend, n * p); zb(1, :) = z';
    for s = 2:Tend
        z = [c * s.^((0:size(c, 2) - 1)'); zeros(n * (p - 1), 1)] + F * z + [res(:, idx(s)); zeros(n * (p - 1), 1)];
        zb(s, :) = z';
    end
    if ~isequal(zb(TBurnIn + 1:Tend, 1:n), Ysim)
        error('var_bootstrap replay mismatch (%s)', tag);
    end
    chk('varboot', Ysim);
    syn_writecsv(fullfile(d, [tag '_input.csv']), Yb);
    syn_writecsv(fullfile(d, [tag '_c.csv']), c);
    syn_writecsv(fullfile(d, [tag '_phi.csv']), reshape(phi, n, n * p));
    syn_writecsv(fullfile(d, [tag '_residual.csv']), res);
    syn_writecsv(fullfile(d, [tag '_idx.csv']), idx);
    syn_writecsv(fullfile(d, [tag '_init.csv']), init);
    syn_writecsv(fullfile(d, [tag '_sim.csv']), Ysim);
    syn_writecsv(fullfile(d, [tag '_meta.csv']), [p, T, TBurnIn, Teff]);
    printf('var_bootstrap %s: p = %d, init = %d, idx(1:5) = %s\n', tag, p, init, mat2str(idx(1:5)'));
end

printf('done\n');
