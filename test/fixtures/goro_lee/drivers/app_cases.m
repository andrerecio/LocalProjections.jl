function [C, raw] = app_cases()
% Application inputs exactly as AppSimulation.m / AppSimulationRZ.m build
% them (Section 5: T = 160, 1969Q1-2008Q4; Appendix H2: T = 384,
% 1920Q1-2015Q4), plus the per-case arguments of the estimator calls.
AppData;                                   % defines `data` (160 x 8)
timeseries = data(1:160, [2, 4, 8, 3, 7]); % TFP, 400*Delta y, pi, MP, r
y   = cumsum(timeseries(:, 2));
ppi = timeseries(:, 3);
XTFP = timeseries;
XMP  = timeseries(:, [4, 1, 2, 3, 5]);
raw.sec5 = data;
raw.sec5_timeseries = timeseries;

AppDataRZ;                                 % defines `macro_quarter` (384 x 6)
tsrz = macro_quarter(1:384, [2, 3, 4, 6]); % news, 400*D lnY, pi, r
yrz   = cumsum(tsrz(:, 2));
pirz  = tsrz(:, 3);
raw.rz = macro_quarter;
raw.rz_timeseries = tsrz;

namesTFP = {'tfp', 'dy', 'pi', 'rr', 'effr'};
namesMP  = {'rr', 'tfp', 'dy', 'pi', 'effr'};
namesRZ  = {'news', 'dy', 'pi', 'tbill'};

C = struct([]);
C = addcase(C, 'tfp_y',  y,    XTFP, namesTFP, 0, [1,3,4,5], 2, [1,0,0,0,0], 1, [1,0,0,0,0]);
C = addcase(C, 'tfp_pi', ppi,  XTFP, namesTFP, 1, [1,2,4,5], 3, [1,0,0,0,0], 1, [1,0,0,0,0]);
C = addcase(C, 'mp_y',   y,    XMP,  namesMP,  0, [2,4,1,5], 2, [1,1,1,1,0], 0, [0,0,0,0,0]);
C = addcase(C, 'mp_pi',  ppi,  XMP,  namesMP,  1, [2,3,1,5], 3, [1,1,1,1,0], 0, [0,0,0,0,0]);
C = addcase(C, 'rz_y',   yrz,  tsrz, namesRZ,  0, [1,3,4],   2, [1,0,0,0],   1, [1,0,0,0]);
C = addcase(C, 'rz_pi',  pirz, tsrz, namesRZ,  1, [1,2,4],   3, [1,0,0,0],   1, [1,0,0,0]);
end

function C = addcase(C, name, y, X, names, ylevel, XVAROrder, yPlace, XtInd, lagyLP, lagXLP)
c.name = name;
c.y = y;
c.X = X;
c.names = names;
c.ylevel = ylevel;
c.lagy = 0;
c.lagX = 4 * ones(1, size(X, 2));
c.trend = 0;
c.horizon = 20;
c.XVAROrder = XVAROrder;
c.yPlace = yPlace;
c.XtIndicator = XtInd;
c.lagyLP = lagyLP;
c.lagXLP = lagXLP;
c.lagVARSet = 1:10;
c.ICIndicator = 1;   % HQIC
c.dfIndicator = 0;
if isempty(C)
    C = c;
else
    C(end+1) = c;
end
end
