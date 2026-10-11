function out = app_varic(y, X, horizon, ylevel, trend, lagVARSet, XVAROrder, yPlace, ICIndicator)
% Deterministic part of BootstrapVARR2.m / BootstrapVARLPB.m, sections
% #2-#3.1 (BootstrapVARR2.m lines 104, 135-173, 178-187, 217-225): the VAR
% system XVAR, the information-criterion lag choice, and the FEVD of the
% bootstrap DGP (vdVAR, the pseudo-truth s_VAR used in vdBc).
% Hand transcription with comments removed: the statements are the
% reference ones, except that (a) only the multi-candidate branch of
% `if isscalar(lagVARSet)` is kept (lagVARSet is always 1:10 here), (b) the
% unused InvXVAROrder block (lines 190-212) is dropped, and (c) var_chol's
% 4th argument (Monte Carlo replications) is 2000 instead of NBootstrap,
% which has no effect because MCIndicator = 0.
[Ty, N] = size(X);
XVAR = X(:, XVAROrder);
if ylevel == 1
    XVAR = [XVAR(:,1:yPlace-1), y, XVAR(:,yPlace:end)];
else
    XVAR = [XVAR(2:end,1:yPlace-1), diff(y), XVAR(2:end,yPlace:end)];
end
NXVAR = size(XVAR,2);

IC = zeros(size(lagVARSet));
lagVARMax = max(lagVARSet);
T = Ty - lagVARMax - (1-ylevel);
for idxIC = 1:length(IC)
    lagVARtemp = lagVARSet(idxIC);
    [~, ~, omega, ~, k] = var_estimation(XVAR(lagVARMax - lagVARtemp + 1:end, :), lagVARtemp, 0, trend);
    if ICIndicator == 0
        IC(idxIC) = log(det(omega)) + k * log(T) / T;
    elseif ICIndicator == 1
        IC(idxIC) = log(det(omega)) + 2 * k * log(log(T)) / T;
    else
        IC(idxIC) = log(det(omega)) + 2 * k / T;
    end
end
[~, idxIC] = min(IC);
lagVAR = lagVARSet(idxIC);

XVAROrder = XVAROrder(:);
XVAROrderwy = [XVAROrder(1:yPlace-1); 0; XVAROrder(yPlace:end)];
OnetoNXVAR = 1:NXVAR;
xLocation = OnetoNXVAR(XVAROrderwy == 1);
yLocation = yPlace;
if ylevel == 0
    differencedIndex = yLocation;
else
    differencedIndex = [];
end
[~, ~, ~, vdVAR, ~, ~, ~, c, phi, omegaDGP, ~, ~, TDGP, residual] ...
    = var_chol(XVAR, lagVAR, horizon, 2000, xLocation, 0, differencedIndex, trend, 0);

out.XVAR = XVAR;
out.IC = IC(:);
out.T_IC = T;
out.lagVAR = lagVAR;
out.xLocation = xLocation;
out.yLocation = yLocation;
out.vdVAR = vdVAR(yLocation, :)';
out.vdVARall = vdVAR;
out.c = c;
out.phi = phi;
out.omega = omegaDGP;
out.TDGP = TDGP;
end
